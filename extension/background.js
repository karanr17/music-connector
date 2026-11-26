const OFFSCREEN_URL = 'offscreen.html';
const SUPPORTED = /^https:\/\/(music|www)\.youtube\.com\//;

chrome.action.onClicked.addListener(async (tab) => {
  const { tabId } = await chrome.storage.session.get('tabId');
  if (tabId === tab.id) {
    await stopCapture();
    return;
  }
  if (!SUPPORTED.test(tab.url || '')) {
    await setBadge(tab.id, '!', '#b3261e', 'Open music.youtube.com or youtube.com first');
    return;
  }
  if (tabId !== undefined) await stopCapture();
  await startCapture(tab);
});

chrome.tabs.onRemoved.addListener(async (closedId) => {
  const { tabId } = await chrome.storage.session.get('tabId');
  if (closedId === tabId) await stopCapture();
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.target !== 'background') return;
  if (msg.type === 'need-snapshot') {
    chrome.tabs.sendMessage(msg.tabId, { type: 'snapshot' }).catch(() => {});
  } else if (msg.type === 'status') {
    chrome.storage.session.get('tabId').then(({ tabId }) => {
      if (tabId === undefined) return;
      if (msg.connected) setBadge(tabId, 'ON', '#1b873f', 'Music Connector: capturing (click to stop)');
      else setBadge(tabId, '…', '#6b6b6b', 'Music Connector: waiting for `mc serve` on port 8765');
    });
  }
});

async function startCapture(tab) {
  await ensureContentScripts(tab.id);
  await ensureOffscreen();
  // Must run in direct response to the toolbar click: Chrome only grants tab capture then.
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
  await chrome.storage.session.set({ tabId: tab.id });
  await chrome.runtime.sendMessage({ target: 'offscreen', type: 'start', streamId, tabId: tab.id });
  await chrome.tabs.sendMessage(tab.id, { type: 'capture', on: true }).catch(() => {});
  await setBadge(tab.id, '…', '#6b6b6b', 'Music Connector: connecting to `mc serve`');
}

async function stopCapture() {
  const { tabId } = await chrome.storage.session.get('tabId');
  await chrome.storage.session.remove('tabId');
  if (await chrome.offscreen.hasDocument()) {
    await chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop' }).catch(() => {});
  }
  if (tabId !== undefined) {
    chrome.tabs.sendMessage(tabId, { type: 'capture', on: false }).catch(() => {});
    await setBadge(tabId, '', null, 'Music Connector: click to start capturing this tab');
  }
}

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['USER_MEDIA'],
    justification: 'Capture tab audio to embed songs locally',
  });
}

// Tabs opened before the extension was installed or reloaded don't have the content scripts.
async function ensureContentScripts(tabId) {
  const alive = await chrome.tabs.sendMessage(tabId, { type: 'ping' }).catch(() => false);
  if (alive) return;
  await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
  await chrome.scripting.executeScript({ target: { tabId }, files: ['page.js'], world: 'MAIN' });
}

async function setBadge(tabId, text, color, title) {
  await chrome.action.setBadgeText({ tabId, text }).catch(() => {});
  if (color) await chrome.action.setBadgeBackgroundColor({ tabId, color }).catch(() => {});
  await chrome.action.setTitle({ tabId, title }).catch(() => {});
}
