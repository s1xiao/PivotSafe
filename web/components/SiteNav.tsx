import Link from "next/link";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "总览" },
  { href: "/detect", label: "文本检测" },
  { href: "/article-detect", label: "文章检测" },
  { href: "/image-detect", label: "图片检测" },
  { href: "/video-detect", label: "视频检测" },
  { href: "/batch", label: "批量处理" },
  { href: "/redteam", label: "红队攻击" },
  { href: "/vuln", label: "脆弱性分析" },
];

export function SiteNav({ path }: { path: string }) {
  return (
    <header className="border-b border-zinc-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-3">
        <Link href="/" className="font-semibold text-zinc-900 tracking-tight">
          CCCC 多模态攻防
        </Link>
        <nav className="flex flex-wrap gap-1 text-sm">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                "rounded-md px-3 py-1.5 transition-colors",
                path === l.href
                  ? "bg-zinc-900 text-white"
                  : "text-zinc-600 hover:bg-zinc-100"
              )}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}


