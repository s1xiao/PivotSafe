"use client";

import { usePathname } from "next/navigation";
import { SiteNav } from "@/components/SiteNav";

export function NavShell({ children }: { children: React.ReactNode }) {
  const path = usePathname() || "/";
  return (
    <>
      <SiteNav path={path} />
      <div className="mx-auto max-w-5xl px-4 py-8">{children}</div>
    </>
  );
}
