// @mkkellogg/gaussian-splats-3d ships no TypeScript declarations (no .d.ts,
// no "types" field in package.json) -- this ambient declaration treats the
// whole module as `any` so imports type-check under strict mode without us
// hand-writing full types for a third-party library we only call a few
// methods on.
declare module "@mkkellogg/gaussian-splats-3d";
