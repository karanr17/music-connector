// Main-world script: reads YouTube's own player API (#movie_player.getVideoData etc.), which
// stays correct on YT Music pages whose URL doesn't name the playing song.
(() => {
  if (window.__musicConnectorPage) return;
  window.__musicConnectorPage = true;

  const SOURCE = location.hostname === 'music.youtube.com' ? 'ytmusic' : 'youtube';
  const POLL_MS = 500;
  let last = {};

  const post = (msg) => window.postMessage({ __musicConnector: true, ...msg }, location.origin);

  function likeStatus() {
    if (SOURCE === 'ytmusic') {
      const el = document.querySelector('ytmusic-player-bar ytmusic-like-button-renderer');
      return el ? el.getAttribute('like-status') : null;
    }
    if (document.querySelector('like-button-view-model button[aria-pressed="true"]')) return 'LIKE';
    if (document.querySelector('dislike-button-view-model button[aria-pressed="true"]')) return 'DISLIKE';
    return 'INDIFFERENT';
  }

  function snapshot() {
    const player = document.getElementById('movie_player');
    const video = player && player.querySelector('video');
    if (!video || typeof player.getVideoData !== 'function') return null;
    const data = player.getVideoData() || {};
    const meta = navigator.mediaSession && navigator.mediaSession.metadata;
    const duration = player.getDuration ? player.getDuration() : video.duration;
    return {
      videoId: data.video_id,
      // getVideoData belongs to the current video; mediaSession can lag a track behind.
      title: data.title || (meta && meta.title),
      artist: data.author || (meta && meta.artist),
      duration: Number.isFinite(duration) && duration > 0 ? duration : null,
      position: video.currentTime,
      playing: !video.paused && !video.ended && video.readyState > 2,
      ad: player.classList.contains('ad-showing'),
      volume: video.muted ? 0 : video.volume,
      like: likeStatus(),
    };
  }

  function tick() {
    const s = snapshot();
    if (!s) return;
    if (s.ad !== last.ad) {
      post({ type: 'ad', active: s.ad });
      last.ad = s.ad;
    }
    // During an ad getVideoData describes the ad, not the song.
    if (s.ad || !s.videoId) return;

    if (s.videoId !== last.videoId) {
      last = { ad: last.ad, videoId: s.videoId, like: s.like };
      post({ type: 'track', videoId: s.videoId, title: s.title, artist: s.artist,
             duration: s.duration, position: s.position, source: SOURCE });
    }
    if (s.playing !== last.playing || s.duration !== last.duration ||
        Math.abs(s.volume - (last.volume ?? -1)) > 0.01) {
      post({ type: 'state', playing: s.playing, position: s.position, volume: s.volume,
             duration: s.duration });
      Object.assign(last, { playing: s.playing, volume: s.volume, duration: s.duration });
    }
    // Only changes count: a song you liked months ago shows LIKE every time it plays.
    if (s.like !== last.like) {
      if (s.like === 'LIKE' || s.like === 'DISLIKE') {
        post({ type: 'feedback', videoId: s.videoId, kind: s.like.toLowerCase(),
               title: s.title, artist: s.artist, source: SOURCE });
      }
      last.like = s.like;
    }
  }

  // Media events don't bubble, but a capturing listener on document still sees them.
  for (const name of ['play', 'playing', 'pause', 'waiting', 'volumechange', 'loadedmetadata']) {
    document.addEventListener(name, tick, true);
  }
  document.addEventListener('seeked', (e) => {
    if (e.target.closest && e.target.closest('#movie_player') && !last.ad) {
      tick();
      post({ type: 'seek', position: e.target.currentTime });
    }
  }, true);
  document.addEventListener('ended', (e) => {
    if (e.target.closest && e.target.closest('#movie_player') && !last.ad) {
      post({ type: 'end', reason: 'ended' });
      last.videoId = null;
    }
  }, true);

  window.addEventListener('message', (e) => {
    if (e.source === window && e.data && e.data.__musicConnectorCmd === 'snapshot') {
      last = {};
      tick();
    }
  });

  setInterval(tick, POLL_MS);
})();
