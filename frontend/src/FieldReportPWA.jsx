/**
 * FieldReportPWA.jsx
 *
 * Geo-tagged hazard report form for ground officers, built to work in poor
 * or no connectivity: it captures the officer's location automatically,
 * lets them pick a hazard type, add notes, and attach a photo, then either
 * POSTs immediately (online) or queues the report in IndexedDB and syncs it
 * later (offline).
 *
 * Peer dependencies (host project must install):
 *   react, react-dom
 *   idb@^8            -- promise-based IndexedDB wrapper
 *   Tailwind CSS configured in the host project
 *
 * Backend contract: POSTs multipart/form-data to POST /api/submit-report
 * with fields hazardType, notes, latitude, longitude, accuracy, capturedAt,
 * and photo (file). Point SUBMIT_ENDPOINT below at your real API.
 *
 * --------------------------------------------------------------------------
 * On "register a sync event" -- what this actually needs, and the fallback
 * --------------------------------------------------------------------------
 * The Background Sync API (`ServiceWorkerRegistration.sync.register(...)`)
 * lets a service worker flush the queue later even if the officer has
 * closed the tab -- but it ONLY works if the app has already registered a
 * service worker, and it isn't supported at all in Safari/iOS (a very real
 * share of field devices). This component calls `registerBackgroundSync()`
 * below, which:
 *   1. Feature-detects `serviceWorker` + `SyncManager` and, if present,
 *      registers a 'sync-field-reports' tag against whatever service worker
 *      the host app has active. See field-report-sync-worker.js (shipped
 *      alongside this component) for the worker-side handler that actually
 *      flushes IndexedDB to the API when that tag fires -- register that
 *      worker once at your app's entry point, e.g.:
 *        navigator.serviceWorker.register('/field-report-sync-worker.js')
 *      This component does not register it itself, since service-worker
 *      registration is an app-level concern, not a single form's.
 *   2. Regardless of whether that succeeds, ALSO relies on a plain
 *      `window.addEventListener('online', ...)` handler in this component
 *      that flushes the queue itself whenever the tab is open and
 *      connectivity returns. This is what actually guarantees delivery on
 *      Safari/iOS and in the common case where the tab is open when
 *      connectivity comes back -- Background Sync is a best-effort bonus
 *      on top, not the only mechanism.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { openDB } from 'idb'

const SUBMIT_ENDPOINT = '/api/submit-report'
const DB_NAME = 'field-reports-db'
const STORE_NAME = 'reports'
const SYNC_TAG = 'sync-field-reports'

const HAZARD_TYPES = ['Surface Crack', 'Rockfall', 'Blocked Road']

// --------------------------------------------------------------------------
// IndexedDB queue (via idb)
// --------------------------------------------------------------------------

let dbPromise = null
function getDB() {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, 1, {
      upgrade(db) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' })
        store.createIndex('status', 'status')
      },
    })
  }
  return dbPromise
}

async function queueReport(report) {
  const db = await getDB()
  await db.put(STORE_NAME, report)
}

async function getPendingReports() {
  const db = await getDB()
  return db.getAllFromIndex(STORE_NAME, 'status', 'pending')
}

async function countPendingReports() {
  const db = await getDB()
  return db.countFromIndex(STORE_NAME, 'status', 'pending')
}

async function markReportStatus(id, status, extra = {}) {
  const db = await getDB()
  const existing = await db.get(STORE_NAME, id)
  if (!existing) return
  await db.put(STORE_NAME, { ...existing, status, ...extra })
}

async function deleteReport(id) {
  const db = await getDB()
  await db.delete(STORE_NAME, id)
}

// --------------------------------------------------------------------------
// Networking
// --------------------------------------------------------------------------

async function submitReportToServer(report) {
  const form = new FormData()
  form.append('hazardType', report.hazardType)
  form.append('notes', report.notes || '')
  form.append('latitude', report.latitude ?? '')
  form.append('longitude', report.longitude ?? '')
  form.append('accuracy', report.accuracy ?? '')
  form.append('capturedAt', report.capturedAt)
  if (report.photo) {
    form.append('photo', report.photo, report.photoName || 'hazard-photo.jpg')
  }

  const res = await fetch(SUBMIT_ENDPOINT, { method: 'POST', body: form })
  if (!res.ok) {
    throw new Error(`Server responded ${res.status}`)
  }
  return res.json().catch(() => ({}))
}

async function registerBackgroundSync() {
  if (!('serviceWorker' in navigator) || !('SyncManager' in window)) {
    return false
  }
  try {
    // `navigator.serviceWorker.ready` never resolves if no service worker
    // has been registered for this origin (e.g. the host app hasn't wired
    // up field-report-sync-worker.js yet, or registration silently failed)
    // -- without a timeout that hangs this call, and therefore the whole
    // submit flow, forever. This is a best-effort enhancement, not the
    // primary delivery mechanism (see the module docstring), so it must
    // never block the UI.
    const registration = await Promise.race([
      navigator.serviceWorker.ready,
      new Promise((_, reject) => setTimeout(() => reject(new Error('service worker not ready (timed out)')), 3000)),
    ])
    await registration.sync.register(SYNC_TAG)
    return true
  } catch (err) {
    console.warn('[FieldReportPWA] Background sync registration failed or timed out:', err)
    return false
  }
}

// --------------------------------------------------------------------------
// Geolocation hook
// --------------------------------------------------------------------------

function useGeolocation() {
  const [state, setState] = useState({ status: 'idle', position: null, error: null })

  const capture = useCallback(() => {
    if (!('geolocation' in navigator)) {
      setState({ status: 'error', position: null, error: 'Geolocation is not supported on this device.' })
      return
    }
    setState((s) => ({ ...s, status: 'locating', error: null }))
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setState({
          status: 'success',
          error: null,
          position: {
            latitude: pos.coords.latitude,
            longitude: pos.coords.longitude,
            accuracy: pos.coords.accuracy,
            capturedAt: new Date(pos.timestamp).toISOString(),
          },
        })
      },
      (err) => {
        const messages = {
          1: 'Location permission denied. Enable it in your browser/app settings.',
          2: 'Location unavailable right now. Try again in the open air.',
          3: 'Location request timed out. Try again.',
        }
        setState({ status: 'error', position: null, error: messages[err.code] || err.message })
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 },
    )
  }, [])

  useEffect(() => {
    capture()
  }, [capture])

  return { ...state, retry: capture }
}

// --------------------------------------------------------------------------
// Network status banner
// --------------------------------------------------------------------------

function NetworkBanner({ online, pendingCount, syncing, onSyncNow }) {
  if (online) {
    return (
      <div className="flex items-center justify-between gap-3 bg-emerald-600/90 px-4 py-2 text-sm font-medium text-white">
        <span className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-white" />
          Online
        </span>
        {pendingCount > 0 && (
          <button
            type="button"
            onClick={onSyncNow}
            disabled={syncing}
            className="rounded-md bg-white/15 px-2 py-1 text-xs font-semibold hover:bg-white/25 disabled:opacity-60"
          >
            {syncing ? 'Syncing…' : `Sync ${pendingCount} queued report${pendingCount === 1 ? '' : 's'}`}
          </button>
        )}
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 bg-amber-600/90 px-4 py-2 text-sm font-medium text-white">
      <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
      Offline &ndash; Reports Queued{pendingCount > 0 ? ` (${pendingCount})` : ''}
    </div>
  )
}

// --------------------------------------------------------------------------
// Main component
// --------------------------------------------------------------------------

export default function FieldReportPWA() {
  const geo = useGeolocation()

  const [online, setOnline] = useState(typeof navigator !== 'undefined' ? navigator.onLine : true)
  const [hazardType, setHazardType] = useState('')
  const [notes, setNotes] = useState('')
  const [photoFile, setPhotoFile] = useState(null)
  const [photoPreviewUrl, setPhotoPreviewUrl] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [pendingCount, setPendingCount] = useState(0)
  const [queuedItems, setQueuedItems] = useState([])
  const [statusMessage, setStatusMessage] = useState(null) // { kind: 'success'|'queued'|'error', text }
  const fileInputRef = useRef(null)

  const refreshQueue = useCallback(async () => {
    const [count, items] = await Promise.all([countPendingReports(), getPendingReports()])
    setPendingCount(count)
    setQueuedItems(items.sort((a, b) => new Date(b.capturedAt) - new Date(a.capturedAt)))
  }, [])

  const flushQueue = useCallback(async () => {
    if (!navigator.onLine) return
    setSyncing(true)
    try {
      const pending = await getPendingReports()
      for (const report of pending) {
        if (!navigator.onLine) break // connection dropped mid-flush; stop and leave the rest queued
        try {
          await submitReportToServer(report)
          await deleteReport(report.id)
        } catch (err) {
          await markReportStatus(report.id, 'pending', { lastError: String(err) })
        }
      }
    } finally {
      setSyncing(false)
      await refreshQueue()
    }
  }, [refreshQueue])

  // Network status + auto-flush wiring
  useEffect(() => {
    function handleOnline() {
      setOnline(true)
      flushQueue()
    }
    function handleOffline() {
      setOnline(false)
    }
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [flushQueue])

  // Initial load: show any reports queued from a previous session, and try
  // to flush them right away if we're already online.
  useEffect(() => {
    refreshQueue()
    if (navigator.onLine) flushQueue()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Photo preview object-URL lifecycle
  useEffect(() => {
    if (!photoFile) {
      setPhotoPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(photoFile)
    setPhotoPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [photoFile])

  const canSubmit = useMemo(
    () => Boolean(hazardType) && Boolean(photoFile) && !submitting,
    [hazardType, photoFile, submitting],
  )

  function handlePhotoChange(e) {
    const file = e.target.files?.[0]
    if (file) setPhotoFile(file)
  }

  function resetForm() {
    setHazardType('')
    setNotes('')
    setPhotoFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!canSubmit) return

    setSubmitting(true)
    setStatusMessage(null)

    const report = {
      id: crypto.randomUUID(),
      hazardType,
      notes: notes.trim(),
      photo: photoFile,
      photoName: photoFile?.name || 'hazard-photo.jpg',
      latitude: geo.position?.latitude ?? null,
      longitude: geo.position?.longitude ?? null,
      accuracy: geo.position?.accuracy ?? null,
      capturedAt: geo.position?.capturedAt || new Date().toISOString(),
      status: 'pending',
    }

    try {
      if (navigator.onLine) {
        await submitReportToServer(report)
        setStatusMessage({ kind: 'success', text: 'Report submitted successfully.' })
        resetForm()
      } else {
        throw new Error('offline')
      }
    } catch (err) {
      // Covers both "we were offline" and "we were online but the request
      // failed" (flaky connection, server hiccup, etc.) -- either way the
      // report is not lost, it goes in the local queue.
      await queueReport(report)
      await registerBackgroundSync()
      await refreshQueue()
      setStatusMessage({
        kind: 'queued',
        text: navigator.onLine
          ? 'Could not reach the server -- report saved and queued for retry.'
          : 'You are offline -- report saved locally and will sync automatically.',
      })
      resetForm()
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-md flex-col bg-slate-950 text-slate-100">
      <NetworkBanner online={online} pendingCount={pendingCount} syncing={syncing} onSyncNow={flushQueue} />

      <header className="border-b border-slate-800 px-4 py-3">
        <h1 className="text-base font-semibold text-slate-50">Submit Field Hazard Report</h1>
        <p className="text-xs text-slate-500">Photos and location are attached automatically where possible.</p>
      </header>

      <form onSubmit={handleSubmit} className="flex flex-1 flex-col gap-4 px-4 py-4">
        {/* Location */}
        <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Location</span>
            {geo.status !== 'locating' && (
              <button type="button" onClick={geo.retry} className="text-xs font-medium text-sky-400 hover:text-sky-300">
                {geo.status === 'success' ? 'Refresh' : 'Retry'}
              </button>
            )}
          </div>
          {geo.status === 'locating' && <p className="mt-1 text-sm text-slate-400">Getting your location…</p>}
          {geo.status === 'success' && geo.position && (
            <p className="mt-1 font-mono text-sm text-slate-200">
              {geo.position.latitude.toFixed(5)}, {geo.position.longitude.toFixed(5)}
              <span className="ml-2 text-xs text-slate-500">&plusmn;{Math.round(geo.position.accuracy)}m</span>
            </p>
          )}
          {geo.status === 'error' && <p className="mt-1 text-sm text-amber-400">{geo.error}</p>}
        </section>

        {/* Hazard type */}
        <section>
          <label htmlFor="hazardType" className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-400">
            Hazard Type
          </label>
          <select
            id="hazardType"
            value={hazardType}
            onChange={(e) => setHazardType(e.target.value)}
            required
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
          >
            <option value="" disabled>
              Select a hazard type&hellip;
            </option>
            {HAZARD_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </section>

        {/* Notes */}
        <section>
          <label htmlFor="notes" className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-400">
            Notes
          </label>
          <textarea
            id="notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder="Describe what you're seeing…"
            className="w-full resize-none rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none"
          />
        </section>

        {/* Photo */}
        <section>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-400">Photo</label>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handlePhotoChange}
            className="hidden"
            id="photoInput"
          />
          {photoPreviewUrl ? (
            <div className="relative overflow-hidden rounded-lg border border-slate-700">
              <img src={photoPreviewUrl} alt="Hazard preview" className="h-40 w-full object-cover" />
              <button
                type="button"
                onClick={() => setPhotoFile(null)}
                className="absolute right-2 top-2 rounded-full bg-slate-950/80 px-2 py-1 text-xs font-medium text-slate-200 hover:bg-slate-950"
              >
                Remove
              </button>
            </div>
          ) : (
            <label
              htmlFor="photoInput"
              className="flex h-32 cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-slate-700 text-slate-500 hover:border-sky-500 hover:text-sky-400"
            >
              <span className="text-2xl">📷</span>
              <span className="text-sm">Capture or upload a photo</span>
            </label>
          )}
        </section>

        {statusMessage && (
          <div
            className={`rounded-lg px-3 py-2 text-sm ${
              statusMessage.kind === 'success'
                ? 'bg-emerald-500/15 text-emerald-300'
                : statusMessage.kind === 'queued'
                  ? 'bg-amber-500/15 text-amber-300'
                  : 'bg-red-500/15 text-red-300'
            }`}
          >
            {statusMessage.text}
          </div>
        )}

        <button
          type="submit"
          disabled={!canSubmit}
          className="mt-auto w-full rounded-lg bg-sky-600 px-4 py-3 text-sm font-semibold text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
        >
          {submitting ? 'Submitting…' : online ? 'Submit Report' : 'Save Report Offline'}
        </button>
      </form>

      {/* Queued reports */}
      {queuedItems.length > 0 && (
        <section className="border-t border-slate-800 px-4 py-3">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Queued Reports ({queuedItems.length})
          </h2>
          <ul className="space-y-2">
            {queuedItems.map((item) => (
              <li key={item.id} className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2 text-sm">
                <div>
                  <p className="font-medium text-slate-200">{item.hazardType}</p>
                  <p className="text-xs text-slate-500">
                    {new Date(item.capturedAt).toLocaleString()}
                    {item.lastError ? ' · retry failed, will try again' : ''}
                  </p>
                </div>
                <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-300">
                  Pending
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
