// Owns the tab audio stream and the WebSocket to `mc serve`. Offscreen documents only get
// chrome.runtime, so anything involving the tab goes through the service worker.
const SERVER = 'ws://127.0.0.1:8765';
const RECONNECT_MS = 3000;

let tabId = null;
let stream = null;
let ctx = null;
let ws = null;
let reconnectTimer = null;

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.target !== 'offscreen') return;
  if (msg.type === 'start') start(msg.streamId, msg.tabId);
  else if (msg.type === 'stop') stop();
  else if (sender.tab && sender.tab.id === tabId) {
    const { target, ...event } = msg;
    sendJson(event);
  }
});

async function start(streamId, id) {
  await stop();
  tabId = id;
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId } },
    video: false,
  });
  ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  // Capturing a tab silences it; routing the stream back to the speakers keeps it audible.
  source.connect(ctx.destination);

  await ctx.audioWorklet.addModule('capture-worklet.js');
  const node = new AudioWorkletNode(ctx, 'mc-capture');
  const mute = new GainNode(ctx, { gain: 0 });
  // Chrome only runs worklet nodes that are pulled by the destination.
  source.connect(node).connect(mute).connect(ctx.destination);
  node.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  connect();
}

async function stop() {
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  if (ws) {
    sendJson({ type: 'end', reason: 'closed' });
    ws.onclose = null;
    ws.close();
    ws = null;
  }
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (ctx) await ctx.close();
  stream = ctx = null;
  tabId = null;
}

function connect() {
  ws = new WebSocket(SERVER);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    sendJson({ type: 'hello', sampleRate: ctx.sampleRate });
    notify({ type: 'status', connected: true });
    notify({ type: 'need-snapshot', tabId });
  };
  ws.onclose = () => {
    notify({ type: 'status', connected: false });
    if (tabId !== null) reconnectTimer = setTimeout(connect, RECONNECT_MS);
  };
  ws.onerror = () => {};
}

function sendJson(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function notify(msg) {
  chrome.runtime.sendMessage({ target: 'background', ...msg }).catch(() => {});
}
