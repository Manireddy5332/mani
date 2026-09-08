export default function AdminLoading() {
  return (
    <div aria-busy="true" aria-label="Loading administration content">
      <div className="animate-pulse border-b border-line pb-7 motion-reduce:animate-none">
        <div className="h-3 w-28 rounded-full bg-ink/10" />
        <div className="mt-5 h-12 max-w-lg rounded-xl bg-ink/10" />
        <div className="mt-4 h-5 max-w-2xl rounded-full bg-ink/[0.07]" />
      </div>
      <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {Array.from({ length: 6 }).map((_, index) => (
          <div
            key={index}
            className="h-40 animate-pulse rounded-2xl border border-line bg-surface motion-reduce:animate-none"
          />
        ))}
      </div>
    </div>
  );
}
