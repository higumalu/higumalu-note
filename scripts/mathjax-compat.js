// hexo-renderer-kramed handles _ inside $$...$$ correctly.
// This script does minimal cleanup just in case.
'use strict';

hexo.extend.filter.register('after_render:html', function(content) {
  if (!content || !content.includes('$$')) return content;
  // Remove any remaining <em>/</em> from math blocks (conservative)
  content = content.replace(/\$\$([\s\S]*?)\$\$/g, function(block) {
    return '$$' + block.slice(2, -2).replace(/<\/?em>/g, '') + '$$';
  });
  return content;
}, 4);