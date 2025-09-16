import Image from 'next/image'

export default function Hero(){
  return (
    <section aria-label="Introduction" className="py-12">
      <div className="container grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
        <div className="animate-fade-up">
          <h1 className="text-4xl md:text-5xl font-extrabold">Hi, I'm Designer Developer</h1>
          <p className="mt-4 text-lg text-slate-700">I build beautiful interfaces and delightful user experiences. This portfolio is generated from a Figma template.</p>
          <div className="mt-6 flex gap-4">
            <a href="#projects" className="px-4 py-2 rounded bg-accent text-white">View Projects</a>
            <a href="#contact" className="px-4 py-2 rounded border">Contact</a>
          </div>
        </div>
        <div className="flex justify-center md:justify-end animate-fade-up">
          <div className="w-64 h-64 relative">
            <Image src="/Me.png" alt="Me" fill style={{objectFit:'contain'}} />
          </div>
        </div>
      </div>
    </section>
  )
}
