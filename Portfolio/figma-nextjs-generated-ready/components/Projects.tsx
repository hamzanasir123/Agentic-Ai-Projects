import Image from 'next/image'

export default function Projects(){
  const projects = [
    {title:'Project One', img:'/Portfolio.png', desc:'A short description.'},
    {title:'Project Two', img:'/Portfolio-1.png', desc:'A short description.'},
    {title:'Project Three', img:'/Portfolio-2.png', desc:'A short description.'},
  ]
  return (
    <section id="projects" className="py-12 bg-slate-50">
      <div className="container">
        <h2 className="text-2xl font-bold animate-fade-up">Selected Projects</h2>
        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((p)=>(
            <article key={p.title} className="rounded-lg overflow-hidden shadow-sm bg-white" aria-labelledby={p.title}>
              <div className="relative h-48 w-full">
                <Image src={p.img} alt={p.title} fill style={{objectFit:'cover'}} />
              </div>
              <div className="p-4">
                <h3 id={p.title} className="font-semibold">{p.title}</h3>
                <p className="mt-2 text-sm text-slate-600">{p.desc}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
