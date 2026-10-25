import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

const sizeRow = v.object({
  label: v.string(),
  bustMin: v.union(v.number(), v.null()),
  bustMax: v.union(v.number(), v.null()),
  waistMin: v.union(v.number(), v.null()),
  waistMax: v.union(v.number(), v.null()),
  hipsMin: v.union(v.number(), v.null()),
  hipsMax: v.union(v.number(), v.null()),
})

export default defineSchema({
  booth_scans: defineTable({
    height: v.string(),
    timestamp: v.number(),
  }),
  products: defineTable({
    productId: v.string(),
    name: v.string(),
    brand: v.string(),
    url: v.string(),
    image: v.string(),
    price: v.string(),
    areas: v.array(v.string()),
    fitNote: v.string(),
    inseamIn: v.union(v.number(), v.null()),
    rows: v.array(sizeRow),
  }).index("by_productId", ["productId"]),
});
