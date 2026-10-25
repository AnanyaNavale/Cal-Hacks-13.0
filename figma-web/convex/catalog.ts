import { mutation, query } from "./_generated/server";
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

const product = v.object({
  id: v.string(),
  name: v.string(),
  brand: v.string(),
  url: v.string(),
  image: v.string(),
  price: v.string(),
  areas: v.array(v.string()),
  fitNote: v.string(),
  inseamIn: v.union(v.number(), v.null()),
  rows: v.array(sizeRow),
})

export const list = query({
  args: {},
  handler: async (ctx) => {
    const products = await ctx.db.query("products").collect()
    return products.map(product => ({
      id: product.productId,
      name: product.name,
      brand: product.brand,
      url: product.url,
      image: product.image,
      price: product.price,
      areas: product.areas,
      fitNote: product.fitNote,
      inseamIn: product.inseamIn,
      rows: product.rows,
    }))
  },
})

export const replace = mutation({
  args: { products: v.array(product) },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("products").collect()
    for (const row of existing) await ctx.db.delete(row._id)
    for (const item of args.products) {
      await ctx.db.insert("products", {
        productId: item.id,
        name: item.name,
        brand: item.brand,
        url: item.url,
        image: item.image,
        price: item.price,
        areas: item.areas,
        fitNote: item.fitNote,
        inseamIn: item.inseamIn,
        rows: item.rows,
      })
    }
  },
})
