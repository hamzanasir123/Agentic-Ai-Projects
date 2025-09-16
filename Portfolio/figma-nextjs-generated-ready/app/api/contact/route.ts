import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const dataDir = path.join(process.cwd(), 'data')
    await fs.mkdir(dataDir, { recursive: true })
    const file = path.join(dataDir, 'contacts.json')
    let arr = []
    try {
      const existing = await fs.readFile(file, 'utf8')
      arr = JSON.parse(existing)
    } catch (err) {
      arr = []
    }
    const entry = { id: Date.now(), ...body, createdAt: new Date().toISOString() }
    arr.push(entry)
    await fs.writeFile(file, JSON.stringify(arr, null, 2), 'utf8')
    return NextResponse.json({ ok: true })
  } catch (err) {
    return NextResponse.json({ ok: false }, { status: 500 })
  }
}
