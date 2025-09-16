'use client'
import Link from 'next/link'
import { useState, useEffect } from 'react'

export default function Header(){
  const [open, setOpen] = useState(false)

  // close on resize to bigger screens
  useEffect(()=>{
    function onResize(){ if (window.innerWidth >= 768) setOpen(false) }
    window.addEventListener('resize', onResize)
    return ()=>window.removeEventListener('resize', onResize)
  },[])

  return (
    <header className="bg-white shadow-sm sticky top-0 z-40">
      <div className="container flex items-center justify-between py-4">
        <Link href="/" className="text-xl font-bold" aria-label="Home">Designer Developer</Link>

        <nav className="hidden md:flex gap-6" aria-label="Primary">
          <a href="#about" className="hover:underline">About</a>
          <a href="#projects" className="hover:underline">Projects</a>
          <a href="#contact" className="hover:underline">Contact</a>
        </nav>

        <div className="md:hidden">
          <button aria-expanded={open} aria-label="Toggle menu" onClick={()=>setOpen(!open)}
            className="p-2 rounded focus:outline-none focus:ring-2">
            <span>{open ? '✕' : '☰'}</span>
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <div className={`md:hidden transition-max-h duration-300 overflow-hidden ${open ? 'max-h-40' : 'max-h-0'}`}>
        <nav className="px-4 pb-4 flex flex-col gap-2" aria-label="Mobile">
          <a href="#about" onClick={()=>setOpen(false)} className="block py-2">About</a>
          <a href="#projects" onClick={()=>setOpen(false)} className="block py-2">Projects</a>
          <a href="#contact" onClick={()=>setOpen(false)} className="block py-2">Contact</a>
        </nav>
      </div>
    </header>
  )
}
