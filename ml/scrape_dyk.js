(async function extractFromAllLinkedPages() {
  console.log('🔍 Searching for links in <tr><td><a>...');

  function downloadFile(content, fileName, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }

  function cleanDYKLine(line) {
    let cleaned = line.replace(/^\*\s+/, '');
    
    cleaned = cleaned.replace(/''\(pictured\)''/g, '');
    cleaned = cleaned.replace(/\(pictured\)/g, '');
    cleaned = cleaned.replace(/^(ALT\d+:\s*)/, '');
    cleaned = cleaned.replace(/''/g, '');

    cleaned = cleaned.replace(/\[\[([^\]]+)\]\]/g, (match, content) => {
      if (content.includes('|')) {
        const parts = content.split('|');
        return parts[parts.length - 1];
      } else {
        return content;
      }
    });

    const sourceIndex = cleaned.indexOf(' Source:');
    if (sourceIndex !== -1) {
      cleaned = cleaned.substring(0, sourceIndex);
    }

    cleaned = cleaned.replace(/…/g, '...');
    cleaned = cleaned.replace(/\s+,/g, ',');
    cleaned = cleaned.replace(/^\*\s+/, '');
    cleaned = cleaned.trim();

    return cleaned;
  }

  const linkElements = document.querySelectorAll('tr td a');
  const urls = [];
  linkElements.forEach(a => {
    try {
      const absoluteUrl = new URL(a.href, window.location.href);
      urls.push(absoluteUrl.href);
    } catch (e) {}
  });

  const uniqueUrls = [...new Set(urls)];
  console.log(`Found ${uniqueUrls.length} unique URL addresses.`);

  if (uniqueUrls.length === 0) {
    console.log('No links to process.');
    return;
  }

  async function processPage(url) {
    try {
      console.log(`⏳ Downloading: ${url}`);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const html = await response.text();

      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');

      const lists = [];
      doc.querySelectorAll('div.dyk-img').forEach(div => {
        const next = div.nextElementSibling;
        if (next && next.tagName === 'UL') {
          lists.push(next);
        }
      });

      const listTexts = lists.map(ul => ul.innerText.trim());
      return { url, count: lists.length, lists: listTexts };
    } catch (error) {
      return { url, error: error.message };
    }
  }

  const allResults = [];
  let fullReport = '========== SCRAPING RAPORT ==========\n\n';

  for (const [index, url] of uniqueUrls.entries()) {
    console.log(`📄 Processing ${index + 1} / ${uniqueUrls.length}: ${url}`);
    const result = await processPage(url);
    allResults.push(result);

    if (result.error) {
      const msg = `❌ Error: ${result.url} -> ${result.error}`;
      console.log(msg);
      fullReport += msg + '\n\n';
    } else {
      const header = `✅ ${result.url} (found ${result.count} lists):`;
      console.log(header);
      fullReport += header + '\n';
      result.lists.forEach((text, i) => {
        const line = `   Lista ${i+1}: ${text}`;
        console.log(line);
        fullReport += line + '\n';
      });
      fullReport += '\n';
    }

    await new Promise(resolve => setTimeout(resolve, 300));
  }

  const allRawLines = [];
  allResults.forEach(res => {
    if (!res.error && res.lists) {
      res.lists.forEach(listText => {
        const lines = listText.split('\n');
        lines.forEach(line => {
          line = line.trim();
          if (line.includes('… that') || line.includes('... that')) {
            allRawLines.push(line);
          }
        });
      });
    }
  });

  console.log(`Found ${allRawLines.length} DYK points (before cleaning).`);

  const allCleanLines = allRawLines.map(line => cleanDYKLine(line));

  const cleanReport = allCleanLines.join('\n');

  window.scrapedResults = allResults;
  window.scrapedReport = fullReport;
  window.scrapedCleanLines = allCleanLines;
  window.scrapedCleanReport = cleanReport;

  console.log(`\n✅ Scraping done! Cleaned ${allCleanLines.length} lines.`);

  if (confirm('Do you want to download a file with cleaned DYK points? (.txt)?')) {
    downloadFile(cleanReport, 'clean_DYK.txt', 'text/plain');
    console.log('📥 File downloaded');
  } else {
    console.log('Downloading cancelled. You can copy data from window.scrapedCleanLines or window.scrapedCleanReport.');
  }
})();