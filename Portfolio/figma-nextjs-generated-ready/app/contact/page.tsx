'use client'
import { useState } from 'react'

export default function Contact() {
  const [form, setForm] = useState({name:'', email:'', message:''})
  const [status, setStatus] = useState<'idle'|'loading'|'success'|'error'>('idle')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setStatus('loading')
    try {
      const res = await fetch('/api/contact', {
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(form)
      })
      if (res.ok) {
        setStatus('success')
        setForm({name:'', email:'', message:''})
      } else {
        setStatus('error')
      }
    } catch (err) {
      setStatus('error')
    }
  }

  return (
    <section id="contact" className="py-16">
      <div className="container max-w-xl mx-auto">
        <h2 className="text-2xl font-bold">Contact</h2>
        <p className="mt-2 text-slate-600">Want to work together? Send a message.</p>

        <form onSubmit={handleSubmit} className="mt-6 grid gap-4">
          <label className="flex flex-col">
            <span className="text-sm font-medium">Name</span>
            <input required value={form.name} onChange={(e)=>setForm({...form, name:e.target.value})}
              className="mt-1 p-3 rounded border focus:outline-none focus:ring-2" />
          </label>

          <label className="flex flex-col">
            <span className="text-sm font-medium">Email</span>
            <input required type="email" value={form.email} onChange={(e)=>setForm({...form, email:e.target.value})}
              className="mt-1 p-3 rounded border focus:outline-none focus:ring-2" />
          </label>

          <label className="flex flex-col">
            <span className="text-sm font-medium">Message</span>
            <textarea required value={form.message} onChange={(e)=>setForm({...form, message:e.target.value})}
              className="mt-1 p-3 rounded border focus:outline-none focus:ring-2 min-h-[120px]" />
          </label>

          <div className="flex items-center gap-4">
            <button disabled={status==='loading'} type="submit" className="px-4 py-2 rounded bg-accent text-white">
              {status==='loading' ? 'Sending...' : 'Send Message'}
            </button>
            {status==='success' && <div className="text-green-600">Message sent — thank you!</div>}
            {status==='error' && <div className="text-red-600">Something went wrong. Try again.</div>}
          </div>
        </form>
      </div>
    </section>
  )
}
