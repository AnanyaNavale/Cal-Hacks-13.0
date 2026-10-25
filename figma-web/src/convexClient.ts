import { ConvexReactClient } from 'convex/react'
import { anyApi } from 'convex/server'

export const api = anyApi
export const convexUrl = import.meta.env.VITE_CONVEX_URL as string | undefined
export const convex = convexUrl ? new ConvexReactClient(convexUrl) : null
