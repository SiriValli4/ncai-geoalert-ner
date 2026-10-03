import { act } from 'react'
import { createRoot } from 'react-dom/client'
import useLiveRiskPolygons from './useLiveRiskPolygons'

globalThis.IS_REACT_ACT_ENVIRONMENT = true

const BASE = {
  type: 'FeatureCollection',
  features: [
    {
      type: 'Feature',
      properties: {
        district: 'West Kameng',
        risk_level: 'Low',
        risk_score: 10,
        prediction_input: { latitude: 27.26, longitude: 92.4, rainfall_24h_mm: 80, soil_moisture_percent: 58, slope_degree: 35 },
      },
      geometry: null,
    },
    // No prediction_input: must pass through untouched.
    { type: 'Feature', properties: { district: 'Static', risk_level: 'High', risk_score: 90 }, geometry: null },
  ],
}

const PREDICTION = {
  landslide_susceptibility_index: 65.4,
  severity: 'HIGH',
  top_factors: [{ feature: 'rainfall_24h_mm', explanation: 'heavy rain' }],
  estimated_inputs: ['elevation_m'],
  model_type: 'RandomForestClassifier',
}

const ok = (body) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) })
const down = () => Promise.reject(new TypeError('Failed to fetch'))

let container
let root
let latest

function Harness() {
  latest = useLiveRiskPolygons(BASE, { refreshMs: 0 })
  return null
}

async function render() {
  await act(async () => {
    root.render(<Harness />)
  })
}

async function flush() {
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0))
  })
}

beforeEach(() => {
  container = document.createElement('div')
  root = createRoot(container)
  global.fetch = jest.fn()
})

afterEach(() => {
  act(() => root.unmount())
  delete global.fetch
})

test('live predictions recolour by score band and keep the model class', async () => {
  global.fetch.mockImplementation(() => ok(PREDICTION))
  await render()
  await flush()

  expect(latest.status).toBe('live')
  const [live, untouched] = latest.polygons.features
  expect(live.properties.risk_score).toBe(65)
  // 65 is Medium under the system-wide bands even though the model's class is HIGH.
  expect(live.properties.risk_level).toBe('Medium')
  expect(live.properties.prediction.model_severity).toBe('High')
  expect(live.properties.risk_source).toBe('model')
  expect(untouched).toBe(BASE.features[1])
  expect(global.fetch).toHaveBeenCalledTimes(1)
})

test('backend down on first load falls back to the sample values', async () => {
  global.fetch.mockImplementation(down)
  await render()
  await flush()

  expect(latest.status).toBe('offline')
  expect(latest.lastUpdated).toBeNull()
  const props = latest.polygons.features[0].properties
  expect(props.risk_score).toBe(10)
  expect(props.risk_level).toBe('Low')
  expect(props.risk_source).toBe('sample')
})

test('backend dropping out keeps the last live prediction instead of reverting', async () => {
  global.fetch.mockImplementationOnce(() => ok(PREDICTION)).mockImplementation(down)
  await render()
  await flush()
  expect(latest.status).toBe('live')
  const firstUpdate = latest.lastUpdated

  await act(async () => latest.refresh())
  await flush()

  expect(latest.status).toBe('offline')
  expect(latest.lastUpdated).toBe(firstUpdate)
  const props = latest.polygons.features[0].properties
  expect(props.risk_score).toBe(65)
  expect(props.risk_source).toBe('model')
})
