'use strict';
// Convert kramed-passed $...$ inline math to MathJax <script> tags.
// kramed 0.5.6 only handles block-level $$...$$ math; single $...$ passes through
// as plain text. This filter catches all remaining $...$ pairs and wraps them
// in <script type="math/tex"> (inline, non-display) so MathJax can render them.
hexo.extend.filter.register('after_render:html', function(content) {
  if (!content) return content;
  // Replace $...$ (not $$...$$) with <script type="math/tex">...</script>
  // [^$] ensures we don't match $$ which kramed already converted to display scripts
  return content.replace(/\$([^$\n]+)\$/g, function(match, formula) {
    // Don't convert if it looks like a currency price (contains digits at start/end)
    if (/^\d/.test(formula.trim()) || /\d$/.test(formula.trim())) return match;
    return '<script type="math/tex">' + formula + '</script>';
  });
});