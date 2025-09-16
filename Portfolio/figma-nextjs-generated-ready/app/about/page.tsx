import Image from 'next/image'

export default function About() {
  return (
    <section id="about" className="py-16">
      <div className="container grid grid-cols-1 md:grid-cols-3 gap-8 items-start">
        <div className="md:col-span-1">
          <div className="w-full max-w-sm mx-auto md:mx-0">
            <div className="relative w-64 h-64 mx-auto md:mx-0">
              <Image src="/Me.png" alt="Portrait" fill style={{objectFit:'contain'}} />
            </div>
          </div>
        </div>
        <div className="md:col-span-2">
          <h1 className="text-3xl md:text-4xl font-extrabold">About Me</h1>
          <p className="mt-4 text-lg text-slate-700">
            I'm a designer-developer who loves turning ideas into delightful products. I focus on pixel-perfect UI, accessible interactions, and performant frontends.
          </p>
          <h2 className="mt-6 text-xl font-semibold">Skills</h2>
          <ul className="mt-3 grid grid-cols-2 gap-2 text-sm">
            <li>React / Next.js</li>
            <li>TypeScript</li>
            <li>Tailwind CSS</li>
            <li>Design Systems</li>
            <li>Figma / Prototyping</li>
            <li>Accessibility (a11y)</li>
          </ul>
        </div>
      </div>
    </section>
  )
}
