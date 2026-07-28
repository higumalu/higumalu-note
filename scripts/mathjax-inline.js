'use strict';
/**
 * Pipeline:
 * 1) Clean <em> leftovers that markdown _..._ can leave inside math
 * 2) Convert any remaining raw $...$ to MathJax 2 <script type="math/tex">
 * 3) Inject MathJax 2 (required: kramed emits script[type=math/tex], which
 *    MathJax 3 does NOT process; diaspora.js also uses MathJax.Hub)
 */
hexo.extend.filter.register('after_render:html', function(content) {
  if (!content) return content;

  // === Phase 1: Remove <em> inside math script tags ===
  content = content.replace(
    /<script type="math\/tex(?:;\s*mode=display)?">([\s\S]*?)<\/script>/g,
    function(full) {
      return full.replace(/<\/?em>/g, '');
    }
  );

  // === Phase 2: Convert leftover $formula$ to <script type="math/tex"> ===
  if (content.includes('math/tex') || content.includes('$$')) {
    var result = '';
    var i = 0;
    var len = content.length;
    while (i < len) {
      if (content[i] === '$') {
        var j = i + 1;
        var found = false;
        while (j < len) {
          if (content[j] === '$' && content[j - 1] !== '\\') {
            var inner = content.substring(i + 1, j);
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
            }
            found = true;
            j++;
            break;
          }
          j++;
        }
        if (found) {
          i = j;
          continue;
        }
      }
      result += content[i];
      i++;
    }
    content = result;
  }

  // === Phase 3: Inject MathJax 2 (compatible with script[type=math/tex]) ===
  if (content.includes('math/tex') && !content.includes('MathJax.js')) {
    var bodyEnd = content.toLowerCase().lastIndexOf('</body>');
    if (bodyEnd >= 0) {
      var mathjaxScript = [
        '<script type="text/x-mathjax-config">',
        'MathJax.Hub.Config({',
        '  tex2jax: {',
        '    inlineMath: [["$","$"],["\\\\(","\\\\)"]],',
        '    displayMath: [["$$","$$"],["\\\\[","\\\\]"]],',
        '    processEscapes: true',
        '  },',
        '  TeX: { extensions: ["AMSmath.js","AMSsymbols.js"] },',
        '  showProcessingMessages: false,',
        '  messageStyle: "none"',
        '});',
        '</script>',
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-AMS_CHTML" id="MathJax-script"></script>'
      ].join('\n');
      content = content.slice(0, bodyEnd) + mathjaxScript + content.slice(bodyEnd);
    }
  }

  return content;
});
