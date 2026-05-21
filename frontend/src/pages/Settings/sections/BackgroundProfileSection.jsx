// AI Restyle background profile: one-time selfie onboarding → 5
// Gemini-generated backgrounds → pick one.
import { useEffect, useRef, useState } from 'react';
import { useKeys } from '../../../state/keysStore.js';
import {
  createProfile,
  fetchProfile,
  selectBackground,
  regenerate,
  setProfileId,
  useProfileId,
} from '../../../state/profileStore.js';
import { getApiUrl } from '../../../config.js';
import SectionHeader from './SectionHeader.jsx';

export default function BackgroundProfileSection() {
  const profileId = useProfileId();
  const keys = useKeys();
  const fileRef = useRef(null);
  const [profile, setProfile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!profileId) { setProfile(null); return; }
    let alive = true;
    const tick = async () => {
      try {
        const p = await fetchProfile(profileId);
        if (!alive) return;
        setProfile(p);
        if (p.generation_status === 'pending' || p.generation_status === 'generating') {
          setTimeout(tick, 2000);
        }
      } catch {
        if (alive) setProfile(null);
      }
    };
    tick();
    return () => { alive = false; };
  }, [profileId]);

  async function onUpload(e) {
    setError(null);
    const f = e.target.files?.[0];
    if (!f) return;
    if (!keys.gemini) { setError('Set your Gemini key first.'); return; }
    setBusy(true);
    try {
      await createProfile(f, keys.gemini);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  async function onPick(idx) {
    if (!keys.gemini) { setError('Set your Gemini key first.'); return; }
    try {
      await selectBackground(profileId, idx, keys.gemini);
      setProfile(await fetchProfile(profileId));
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  async function onRegenerate() {
    if (!keys.gemini) { setError('Set your Gemini key first.'); return; }
    setBusy(true);
    try {
      await regenerate(profileId, keys.gemini);
      setProfile(await fetchProfile(profileId));
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  function onReplace() {
    setProfileId(null);
    setProfile(null);
  }

  return (
    <div>
      <SectionHeader
        title="Background Profile"
        description="Upload a selfie once. We generate 5 personalized backgrounds for your shorts; pick one and it sticks."
      />

      {!profileId && (
        <div className="rounded-lg border border-dashed border-border p-6 text-center">
          <p className="text-[13px] text-zinc-400 mb-3">No profile yet.</p>
          <button
            onClick={() => fileRef.current?.click()}
            disabled={busy || !keys.gemini}
            className="px-4 py-2 text-[13px] bg-primary text-white rounded-md disabled:opacity-40"
          >
            {busy ? 'Uploading…' : 'Upload selfie & generate backgrounds'}
          </button>
          <input ref={fileRef} type="file" accept="image/png,image/jpeg" hidden onChange={onUpload} />
          {!keys.gemini && <p className="mt-2 text-[11px] text-yellow-400">Set your Gemini key first.</p>}
        </div>
      )}

      {profileId && (profile?.generation_status === 'pending' || profile?.generation_status === 'generating') && (
        <div className="rounded-lg border border-border p-6 text-center">
          <div className="animate-pulse text-[13px] text-zinc-400">
            Generating 5 backgrounds… (~20-40s)
          </div>
        </div>
      )}

      {profileId && profile?.generation_status === 'failed' && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-4">
          <p className="text-[13px] text-red-400 mb-3">Generation failed. Try a different selfie.</p>
          <button onClick={onReplace} className="px-3 py-1.5 text-[12px] border border-border rounded-md">
            Replace selfie
          </button>
        </div>
      )}

      {profileId && profile?.generation_status === 'ready' && (
        <>
          <div className="grid grid-cols-5 gap-3">
            {profile.backgrounds.map((bg) => (
              <button
                key={bg.idx}
                onClick={() => onPick(bg.idx)}
                className={`relative rounded-md overflow-hidden border-2 transition-colors ${
                  profile.selected_idx === bg.idx ? 'border-primary' : 'border-border hover:border-zinc-600'
                }`}
              >
                <img src={getApiUrl(bg.url)} alt={`Background ${bg.idx}`} className="w-full aspect-[9/16] object-cover" />
                {profile.selected_idx === bg.idx && (
                  <div className="absolute top-1 right-1 bg-primary text-white text-[10px] px-1.5 py-0.5 rounded">★</div>
                )}
              </button>
            ))}
          </div>
          <div className="mt-4 flex gap-2">
            <button onClick={onRegenerate} disabled={busy} className="px-3 py-1.5 text-[12px] border border-border rounded-md disabled:opacity-40">
              {busy ? 'Regenerating…' : 'Regenerate 5'}
            </button>
            <button onClick={onReplace} className="px-3 py-1.5 text-[12px] border border-border rounded-md">
              Replace selfie
            </button>
          </div>
        </>
      )}

      {error && <div className="mt-3 text-[12px] text-red-400" role="alert">{error}</div>}
    </div>
  );
}
