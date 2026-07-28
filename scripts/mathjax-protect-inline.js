'use strict';
/**
 * Kramed only treats $$...$$ as math. Single $...$ is parsed as plain text,
 * so markdown escapes like \{ \} become { } and MathJax never sees braces.
 * Convert $...$ → $$...$$ before markdown (preserving real display $$ blocks
 * and skipping code), then kramed emits proper <script type="math/tex">.
 */
hexo.extend.filter.register('before_post_render', function(data) {
  if (!data.content || !data.content.includes('$')) return data;

  var content = data.content;
  var slots = [];

  function save(token) {
    var i = slots.length;
    slots.push(token);
    return '<!--HEXO_MATH_SLOT_' + i + '-->';
  }

  // 1) Park fenced code / inline code so we don't touch $ inside them
  content = content.replace(/```[\s\S]*?```/g, save);
  content = content.replace(/`[^`\n]+`/g, save);

  // 2) Park existing display math $$...$$
  content = content.replace(/\$\$([\s\S]+?)\$\$/g, function(_, tex) {
    return save('$$' + tex + '$$');
  });

  // 3) Promote remaining inline $...$ to $$...$$ for kramed
  content = content.replace(/\$((?:\\.|[^$\\])+?)\$/g, function(_, tex) {
    var trimmed = tex.trim();
    // Skip pure digit currency leftovers / page numbers
    if (/^\d[\d\s.,]*$/.test(trimmed)) {
      return '$' + tex + '$';
    }
    return '$$' + tex + '$$';
  });

  // 4) Restore parked segments
  content = content.replace(/<!--HEXO_MATH_SLOT_(\d+)-->/g, function(_, i) {
    return slots[+i];
  });

  data.content = content;
  return data;
});
