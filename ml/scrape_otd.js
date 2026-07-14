(async function scrapeAllMonths() {
    const months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ];
    const baseUrl = 'https://en.wikipedia.org/wiki/Wikipedia:Selected_anniversaries/';

    const excludeSelectors = [
        'div.hlist',
        'div.hlist.inline.nowraplinks',
        'div.hlist.otd-footer.noprint',
        'div.hlist.plainlinks.metadata.noprint',
        'div.mw-body-header.vector-page-titlebar.no-font-mode-scale',
        'div.vector-page-toolbar.vector-feature-custom-font-size-clientpref--excluded',
        'div.catlinks',
        'div.mw-footer-container',
        'div.vector-menu-content'
    ];
    const excludeSelector = excludeSelectors.join(', ');

    function extractLiFromHtml(html) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');

        const filteredUls = Array.from(doc.querySelectorAll('ul'))
            .filter(ul => !ul.closest(excludeSelector));

        const contents = filteredUls
            .flatMap(ul => Array.from(ul.querySelectorAll('li')))
            .map(li => li.textContent.trim())
            .map(text => text.replace(/\(pictured\)\s*/g, ''))
            .filter(text => text !== '');

        return contents;
    }

    const allItems = [];

    for (const month of months) {
        const url = baseUrl + month;
        try {
            console.log(`Downloading: ${url}`);
            const response = await fetch(url);
            if (!response.ok) {
                console.warn(`⚠️ Page for ${month} returned status ${response.status} – ignoring.`);
                continue;
            }
            const html = await response.text();
            const items = extractLiFromHtml(html);
            console.log(`✅ For ${month} found ${items.length} elements.`);
            allItems.push(...items);
        } catch (error) {
            console.error(`❌ Error for ${month}:`, error);
        }
    }

    if (allItems.length === 0) {
        console.warn('⚠️ No elements found');
        return;
    }

    const outputText = allItems.join('\n');
    const blob = new Blob([outputText], { type: 'text/plain;charset=utf-8' });
    const urlBlob = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = urlBlob;
    link.download = 'otd.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(urlBlob);

    console.log(`✅ Downloaded ${allItems.length} elements.`);
})();