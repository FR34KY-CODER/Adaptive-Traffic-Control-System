function setupStreamControls(imgId, skelId, srcUrl, opts = {}) {
  const img = document.getElementById(imgId);
  const skel = document.getElementById(skelId);
  const pauseBtn = document.getElementById('pauseBtn');
  const resumeBtn = document.getElementById('resumeBtn');
  const snapBtn = document.getElementById('snapBtn');
  const showPause = opts.showPause !== false;

  // skeleton while first frame loads
  img.addEventListener('load', () => skel.classList.add('hide'));
  img.addEventListener('error', () => skel.classList.remove('hide'));

  if (showPause && pauseBtn && resumeBtn) {
    pauseBtn.onclick = () => {
      img.dataset.src = img.src;
      img.src = ''; // stop fetching
      pauseBtn.disabled = true; resumeBtn.disabled = false;
    };
    resumeBtn.onclick = () => {
      img.src = img.dataset.src || srcUrl;
      pauseBtn.disabled = false; resumeBtn.disabled = true;
    };
  }

  if (snapBtn) {
    snapBtn.onclick = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth || 1280;
      canvas.height = img.naturalHeight || 720;
      const ctx = canvas.getContext('2d');
      // Draw the <img> onto canvas (CORS is same-origin for Flask)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const a = document.createElement('a');
      a.href = canvas.toDataURL('image/png');
      a.download = 'snapshot.png';
      a.click();
    };
  }
}

// ROI copy/reset
document.addEventListener('DOMContentLoaded', () => {
  const copyBtn = document.getElementById('copyRoi');
  const resetBtn = document.getElementById('resetRoi');
  const roiInput = document.getElementById('roiInput');

  if (copyBtn && roiInput) {
    copyBtn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(roiInput.value);
        copyBtn.innerHTML = '<i class="bi bi-clipboard-check"></i> Copied';
        setTimeout(() => copyBtn.innerHTML = '<i class="bi bi-clipboard"></i> Copy', 1200);
      } catch {}
    }
  }
  if (resetBtn && roiInput) {
    resetBtn.onclick = () => {
      roiInput.value = "100,300;540,300;620,430;60,430"; // same as DEFAULT_ROI in backend
    }
  }

  // Faux progress for upload (UI polish; server does normal post)
  const form = document.getElementById('uploadForm');
  if (form) {
    const progressWrap = document.getElementById('progressWrap');
    const progressBar = document.getElementById('progressBar');
    form.addEventListener('submit', () => {
      progressWrap.classList.remove('d-none');
      let w = 5;
      const t = setInterval(() => {
        w = Math.min(w + Math.random()*7, 90);
        progressBar.style.width = w + '%';
      }, 250);
      // allow redirect to complete, then clear
      setTimeout(() => clearInterval(t), 15000);
    });
  }
});