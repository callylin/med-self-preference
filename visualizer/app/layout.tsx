import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Conversation Visualizer",
  description: "Visualize generated medical conversations",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
