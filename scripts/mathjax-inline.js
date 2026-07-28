'use strict';
// Convert $...$ inline math (not $$...$$) to MathJax <script> tags.
// Handles adjacent formulas like $A$ $B$ (split by spaces) correctly.
// Also removes any <em>...</em> remnants inside math expressions.
hexo.extend.filter.register('after_render:html', function(content) {
  if (!content) return content;

  // Remove <em> tags INSIDE math expressions (from markdown _..._ processing)
  // that escaped the earlier compat pass.
  content = content.replace(/<em>([^<]*(?:<(?!\/?em>)[^<]*)*)<\/em>/g, '$1');

  // Replace $formula$ with <script type="math/tex">formula</script>
  // Non-greedy (.+?) but carefully handles adjacent formulas.
  var result = '';
  var i = 0;
  var len = content.length;
  while (i < len) {
    if (content[i] === '$') {
      // Look for closing $, but skip if it's an escaped \$ or part of $$
      var j = i + 1;
      var mathContent = '';
      var found = false;
      while (j < len) {
        if (content[j] === '$' && content[j-1] !== '\\') {
          // Found closing $
          // Avoid matching $$...$$ (display math already handled by kramed)
          var inner = content.substring(i+1, j);
          if (inner.includes('$$') || inner.startsWith('$')) {
            // This is display math or escaped dollar, treat as literal
            result += content[i];
            i++;
            break;
          }
          mathContent = inner;
          found = true;
          j++;
          break;
        }
        j++;
      }
      if (found) {
        // Skip leading/trailing whitespace in math content
        var trimmed = mathContent.trim();
        // Don't convert price-like patterns (starts/ends with digit, no commands)
        if (/^\d[\d\s]*$/.test(trimmed)) {
          result += '$' + mathContent + '$';
        } else {
          result += '<script type="math/tex">' + mathContent + '</script>';
        }
        i = j;
      } else {
        result += content[i];
        i++;
      }
    } else {
      result += content[i];
      i++;
    }
  }
  return result;
});