const Footer = () => (
  <footer className="bg-gray-100 border-t border-gray-200">
    <div className="max-w-7xl mx-auto py-5 px-4 sm:px-6 lg:px-8">
      <p className="text-center text-gray-500 text-sm">
        This system is intended strictly for research and educational purposes.
      </p>
      <p className="text-center text-gray-400 text-xs mt-2">
        &copy; {new Date().getFullYear()} AlzheimerAI Project | Ashapu Mohan
      </p>
    </div>
  </footer>
);

export default Footer;