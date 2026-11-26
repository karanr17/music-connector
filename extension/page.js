// Main-world script: reads YouTube's own player API (#movie_player.getVideoData etc.), which
// stays correct on YT Music pages whose URL doesn't name the playing song.
(() => {
  if (window.__musicConnectorPage) return;
  window.__musicConnectorPage = true;

  const SOURCE = location.hostname === 'music.youtube.com' ? 'ytmusic' : 'youtube';
  const POLL_MS = 500;
  let last = {};

  const post = (msg) => window.postMessage({ __musicConnector: true, ...msg }, location.origin);

  // Returns null while the like button still belongs to the previous track: after an autoplay
  // change it keeps the old song's status for a moment, which would read as a fresh like.
  function likeStatus(videoId) {
    if (SOURCE === 'ytmusic') {
      const el = document.querySelector('ytmusic-player-bar ytmusic-like-button-renderer');
      const target = el && el.data && el.data.target && el.data.target.videoId;
      if (!el || (target && target !== videoId)) return null;
      return el.getAttribute('like-status');
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
    const duration = player.getDuration ? player.getDuration() : video.duration;
    return {
      videoId: data.video_id,
      // Only getVideoData is tied to video_id. On a track change its title/author are empty
      // for a moment while mediaSession and the player bar still show the previous song (or
      // the ad), so falling back to those labels the new track with the old song's metadata.
      title: data.title || null,
      artist: data.author || null,
      duration: Number.isFinite(duration) && duration > 0 ? duration : null,
      position: video.currentTime,
      playing: !video.paused && !video.ended && video.readyState > 2,
      ad: player.classList.contains('ad-showing'),
      volume: video.muted ? 0 : video.volume,
      like: likeStatus(data.video_id),
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
      last = { ad: last.ad, videoId: s.videoId, title: s.title, artist: s.artist };
      post({ type: 'track', videoId: s.videoId, title: s.title, artist: s.artist,
             duration: s.duration, position: s.position, source: SOURCE });
    } else if (s.title && (s.title !== last.title || s.artist !== last.artist)) {
      last.title = s.title;
      last.artist = s.artist;
      post({ type: 'meta', videoId: s.videoId, title: s.title, artist: s.artist });
    }
    if (s.playing !== last.playing || s.duration !== last.duration ||
        Math.abs(s.volume - (last.volume ?? -1)) > 0.01) {
      post({ type: 'state', playing: s.playing, position: s.position, volume: s.volume,
             duration: s.duration });
      Object.assign(last, { playing: s.playing, volume: s.volume, duration: s.duration });
    }
    // Only changes count: a song you liked months ago shows LIKE every time it plays. The
    // first status seen for a track is just the baseline.
    if (s.like !== null && s.like !== last.like) {
      if (last.like !== undefined && (s.like === 'LIKE' || s.like === 'DISLIKE')) {
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
