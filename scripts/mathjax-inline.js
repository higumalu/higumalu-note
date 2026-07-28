'use strict';
// Phase 1: Remove <em> remnants from markdown _..._ inside math expressions
// Phase 2: Convert remaining $...$ inline math to MathJax <script> tags
// Phase 3: Inject MathJax config + typeset call (ensures rendering even if CDN loaded before scripts)
hexo.extend.filter.register('after_render:html', function(content) {
  if (!content) return content;

  // === Phase 1: Remove <em> inside math ===
  content = content.replace(/<em>([^<]*(?:<(?!\/?em>)[^<]*)*)<\/em>/g, '$1');

  // === Phase 2: Convert $formula$ to <script type="math/tex"> ===
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

  // === Phase 3: Ensure MathJax typesets our newly injected scripts ===
  // Inject a small script that calls typesetPromise for any remaining math scripts
  if (content.includes('math/tex')) {
    var headEnd = content.toLowerCase().indexOf('</head>');
    if (headEnd >= 0) {
      var typesetScript = '<script>' +
'if (typeof MathJax !== "undefined") {' +
'  MathJax.startup && MathJax.startup.promise && MathJax.startup.promise.then(function() {' +
'    return MathJax.typesetPromise(document.body);' +
'  }).catch(function(e) { console.error("MathJax typeset error:", e); });' +
'}' +
'</script>';
      content = content.slice(0, headEnd) + typesetScript + content.slice(headEnd);
    }
  }

  return content;
});