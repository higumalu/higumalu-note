'use strict';
// Phase 1: Remove <em> remnants from markdown _..._ inside math expressions
// Phase 2: Convert $...$ inline math to MathJax <script type="math/tex"> tags
// Phase 3: Inject MathJax CDN + startup config + typeset at end of <body>
hexo.extend.filter.register('after_render:html', function(content) {
  if (!content) return content;

  // === Phase 1: Remove <em> inside math ===
  content = content.replace(/<em>([^<]*(?:<(?!\/?em>)[^<]*)*)<\/em>/g, '$1');

  // === Phase 2: Convert $formula$ to <script type="math/tex"> (character-by-character) ===
  if (content.includes('math/tex') || content.includes('$$')) {
    var result = '';
    var i = 0;
    var len = content.length;
    var converted = 0;
    while (i < len) {
      if (content[i] === '$') {
        var j = i + 1;
        var found = false;
        while (j < len) {
          if (content[j] === '$' && content[j-1] !== '\\') {
            var inner = content.substring(i+1, j);
            if (inner.includes('$$') || inner.startsWith('$')) {
              result += content[i];
              i++;
              break;
            }
            var trimmed = inner.trim();
            // Don't convert pure digit sequences (page numbers, years, etc.)
            if (/^\d[\d\s]*$/.test(trimmed)) {
              result += '$' + inner + '$';
            } else {
              result += '<script type="math/tex">' + inner + '</script>';
              converted++;
            }
            found = true;
            j++;
            break;
          }
          j++;
        }
        if (found) { i = j; continue; }
      }
      result += content[i];
      i++;
    }
    content = result;
  }

  // === Phase 3: Inject MathJax CDN + config + typeset at end of <body> ===
  if (content.includes('math/tex')) {
    var bodyEnd = content.toLowerCase().lastIndexOf('</body>');
    if (bodyEnd >= 0) {
      var mathjaxScript = [
        '<script>',
        'window.MathJax = {',
        '  tex: { inlineMath: [["$","$"],["\\\\(","\\\\)"]], displayMath: [["$$","$$"],["\\\\[","\\\\]"]], processEscapes: true },',
        '  options: { enableMenu: false },',
        '  startup: {',
        '    ready: function() {',
        '      MathJax.startup.defaultReady();',
        '      MathJax.startup.promise.then(function() { return MathJax.typesetPromise(); });',
        '    }',
        '  }',
        '};',
        '</script>',
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js" id="MathJax-script"></script>'
      ].join('\n');
      content = content.slice(0, bodyEnd) + mathjaxScript + content.slice(bodyEnd);
    }
  }

  return content;
});