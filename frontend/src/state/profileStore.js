// Per-browser onboarding profile: holds the opaque profile_id in
// localStorage; thin wrappers around the backend onboarding routes.
import { useEffect, useState } from 'react';
import { getApiUrl } from '../config.js';

const KEY = 'aiRestyleProfileId_v1';
const EVENT = 'aiRestyleProfileChanged';

export function getProfileId() {
  return localStorage.getItem(KEY);
}

export function setProfileId(id) {
  if (id) localStorage.setItem(KEY, id);
  else localStorage.removeItem(KEY);
  window.dispatchEvent(new CustomEvent(EVENT));
}

export function useProfileId() {
  const [id, setId] = useState(getProfileId());
  useEffect(() => {
    const handler = () => setId(getProfileId());
    window.addEventListener(EVENT, handler);
    window.addEventListener('storage', handler);
    return () => {
      window.removeEventListener(EVENT, handler);
      window.removeEventListener('storage', handler);
    };
  }, []);
  return id;
}

export async function createProfile(selfieFile, geminiKey) {
  const fd = new FormData();
  fd.append('selfie', selfieFile);
  const res = await fetch(getApiUrl('/api/restyle/profile'), {
    method: 'POST',
    headers: { 'X-Gemini-Key': geminiKey },
    body: fd,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const { profile_id } = await res.json();
  setProfileId(profile_id);
  return profile_id;
}

export async function fetchProfile(profileId) {
  const res = await fetch(getApiUrl(`/api/restyle/profile/${profileId}`));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function selectBackground(profileId, idx, geminiKey) {
  const res = await fetch(getApiUrl(`/api/restyle/profile/${profileId}/select`), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Gemini-Key': geminiKey,
    },
    body: JSON.stringify({ idx }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function regenerate(profileId, geminiKey) {
  const res = await fetch(getApiUrl(`/api/restyle/profile/${profileId}/regenerate`), {
    method: 'POST',
    headers: { 'X-Gemini-Key': geminiKey },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}
