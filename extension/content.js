// Isolated-world bridge: page.js (main world) can read YouTube's player but can't use
// chrome.runtime; this script can, but can't see the player's methods.
(() => {
  if (window.__musicConnectorContent) return;
  window.__musicConnectorContent = true;

  let active = false;
  const requestSnapshot = () =>
    window.postMessage({ __musicConnectorCmd: 'snapshot' }, location.origin);

  window.addEventListener('message', (e) => {
    if (!active || e.source !== window || !e.data || !e.data.__musicConnector) return;
    const { __musicConnector, ...event } = e.data;
    chrome.runtime.sendMessage({ target: 'offscreen', ...event }).catch(() => {});
  });

  chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (msg.type === 'ping') {
      reply(true);
      return;
    }
    if (msg.type === 'capture') {
      active = msg.on;
      if (active) requestSnapshot();
    } else if (msg.type === 'snapshot' && active) {
      requestSnapshot();
    }
  });
})();
