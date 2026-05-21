// AI Restyle Step 2: confirm the user's selected profile background and
// submit the video to /api/restyle. The clean-background detection runs
// server-side; any warning appears in the Review step's log feed.

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useKeys } from '../../../state/keysStore.js';
import { useProfileId, fetchProfile } from '../../../state/profileStore.js';
import { getApiUrl } from '../../../config.js';

export default function Precheck({ wizard }) {
  const profileId = useProfileId();
  const keys = useKeys();
  const [profile, setProfile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!profileId) return;
    fetchProfile(profileId).then(setProfile).catch(() => setProfile(null));
  }, [profileId]);

  if (!profileId) {
    return (
      <div className="p-8 max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Set up your background profile</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          AI Restyle needs a selfie-based background profile before it can run.
        </p>
        <Link
          to="/settings/general/background-profile"
          className="inline-block px-4 py-2 text-[13px] bg-primary text-white rounded-md"
        >
          Go to Settings → Background Profile →
        </Link>
      </div>
    );
  }

  if (!profile) {
    return <div className="p-8 text-[13px] text-zinc-400">Loading profile…</div>;
  }

  if (profile.selected_idx == null) {
    return (
      <div className="p-8 max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Pick a background first</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          Your profile has {profile.generated_count} generated backgrounds but none is selected.
        </p>
        <Link
          to="/settings/general/background-profile"
          className="inline-block px-4 py-2 text-[13px] bg-primary text-white rounded-md"
        >
          Pick one →
        </Link>
      </div>
    );
  }

  const selectedBg = profile.backgrounds.find((b) => b.idx === profile.selected_idx);

  async function start() {
    setError(null);
    if (!keys.gemini) { setError('Set your Gemini key in Settings.'); return; }
    if (!keys.fal)    { setError('Set your fal.ai key in Settings.'); return; }
    const fd = new FormData();
    fd.append('file', wizard.data.file.file);
    fd.append('profile_id', profileId);
    setSubmitting(true);
    try {
      const res = await fetch(getApiUrl('/api/restyle'), {
        method: 'POST',
        headers: { 'X-Gemini-Key': keys.gemini, 'X-Fal-Key': keys.fal },
        body: fd,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const { job_id } = await res.json();
      wizard.setData({
        job: { jobId: job_id, status: 'processing', result: null, progress_pct: 0, logs: [] },
      });
      wizard.next();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-[24px] font-semibold mb-2">Confirm & restyle</h1>
      <p className="text-[13px] text-zinc-400 mb-6">
        Your selected background will be used. The server will check if your source background is
        clean enough; a warning shows in the next step if not (but the job still runs).
      </p>

      <div className="rounded-lg border border-border p-4 mb-6">
        <div className="text-[11px] uppercase tracking-wider text-zinc-500 mb-2">Selected background</div>
        <img src={getApiUrl(selectedBg.url)} alt="" className="w-32 aspect-[9/16] object-cover rounded-md" />
      </div>

      {error && <div className="mb-3 text-[12px] text-red-400" role="alert">{error}</div>}

      <div className="flex items-center justify-between">
        <button onClick={wizard.back} className="px-4 py-2 text-[13px] text-zinc-400">← Back</button>
        <button
          onClick={start}
          disabled={submitting || !keys.fal || !keys.gemini}
          className="px-4 py-2 text-[13px] bg-primary text-white rounded-md disabled:opacity-40"
        >
          {submitting ? 'Starting…' : 'Start restyle →'}
        </button>
      </div>
    </div>
  );
}
