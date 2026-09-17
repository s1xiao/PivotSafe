import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { NavShell } from "@/components/NavShell";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "CCCC · 多模态检测与攻防演示",
  description: "文本 / 文章 / 图片 / 视频检测 + 红队攻击 + 脆弱性分析",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-zinc-50 font-sans antialiased text-zinc-900`}
      >
        <NavShell>{children}</NavShell>
      </body>
    </html>
  );
}


