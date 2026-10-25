import { mutation } from "./_generated/server";
import { v } from "convex/values";

export const logScan = mutation({
  args: { height: v.string() },
  handler: async (ctx, args) => {
    await ctx.db.insert("booth_scans", { height: args.height, timestamp: Date.now() });
  },
});
