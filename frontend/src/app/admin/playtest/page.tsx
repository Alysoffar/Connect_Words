import type { Metadata } from "next";
import { AdminPlaytest } from "./admin-playtest";

export const metadata: Metadata = {
  title: "Internal playtest — LinkWords",
  robots: { index: false, follow: false },
};

export default function AdminPlaytestPage() {
  return <AdminPlaytest />;
}
