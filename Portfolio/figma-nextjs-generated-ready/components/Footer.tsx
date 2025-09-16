export default function Footer(){
  return (
    <footer className="border-t mt-12" role="contentinfo">
      <div className="container py-6 text-center text-sm text-slate-600">
        © {new Date().getFullYear()} Your Name. — Built from a Figma template.
      </div>
    </footer>
  )
}
