import './globals.css'
import type { ReactNode } from 'react'

export const metadata = {
  title: 'Designer Developer — Portfolio',
  description: 'Portfolio generated from a Figma template. Responsive, accessible and production-ready starter.',
  openGraph: {
    title: 'Designer Developer — Portfolio',
    description: 'Portfolio generated from a Figma template.',
    images: ['/og-image.png'],
  },
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com"/>
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true"/>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet"/>
      </head>
      <body>
        <div className="min-h-screen flex flex-col bg-white text-slate-900">
          {children}
        </div>
      </body>
    </html>
  )
}
