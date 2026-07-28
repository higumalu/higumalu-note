'use strict';
// Inline MathJax v3 CDN in every HTML page that contains math formulas.
// Replaces the theme's mathjax.ejs partial injection so we control the script tag.
// Writes: config + async CDN script. No external CDN needed beyond cdnjs.
hexo.extend.filter.register('after_render:html', function(content) {
  if (!content || !content.includes('math/tex')) return content;
  // Check if already injected
  if (content.includes('tex-mml-chtml.js')) return content;
  var headEnd = content.toLowerCase().indexOf('</head>');
  if (headEnd < 0) return content;
  var script = '<script>' +
'window.MathJax={tex:{inlineMath:[[\'$\',\'$\'],[\'\\\\\\(\',\'\\\\\\)\']],displayMath:[[\'$$\',\'$$\'],[\'\\\\\\[\',\'\\\\\\]\']],processEscapes:true},options:{skipHtmlTags:[\'script\',\'noscript\',\'style\',\'textarea\',\'pre\',\'code\']},startup:{ready:function(){MathJax.startup.defaultReady()}}};' +
'</script>' +
'<script id="MathJax-script" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>';
  return content.slice(0, headEnd) + script + content.slice(headEnd);
});