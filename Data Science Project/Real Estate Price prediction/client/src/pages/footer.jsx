import Logo from "../assets/vector.png";
import { Link } from "react-router-dom";

const Footer = () => {
    return (
        <div className="bg-gradient-to-br from-stone-950 via-stone-950 to-stone-900">
            {/* Main Footer Content */}
            <div className="w-full max-w-7xl mx-auto px-4 md:px-6 lg:px-8 py-8 md:py-12">
                <div className="flex flex-col lg:flex-row rounded-xl bg-stone-900/50 backdrop-blur-md shadow-lg overflow-hidden">
                    
                    {/* Logo and Description Section */}
                    <div className="w-full lg:w-80 p-4 md:p-6 text-center lg:text-left">
                        <img src={Logo} className="w-48 md:w-60 lg:w-72 mx-auto lg:mx-0" alt="Logo" />
                        <p className="text-amber-700 font-serif text-sm md:text-base lg:text-lg mt-2 md:-mt-2">
                            Our AI app helps you for decision making, one click at a time
                        </p>
                    </div>

                    {/* Links Sections */}
                    <div className="flex flex-col md:flex-row flex-wrap justify-center lg:justify-start gap-6 md:gap-8 lg:gap-12 p-4 md:p-6 mt-4 md:mt-8 lg:mt-6">
                        
                        {/* About Section */}
                        <div className="flex flex-col items-center md:items-start text-amber-700 font-serif min-w-[120px]">
                            <span className="text-lg md:text-xl font-bold mb-2 md:mb-3 border-b-2 border-amber-600 inline-block">About</span>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Our Story</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Careers</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Our Team</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Resources</Link>
                        </div>

                        {/* Support Section */}
                        <div className="flex flex-col items-center md:items-start text-amber-700 font-serif min-w-[120px]">
                            <span className="text-lg md:text-xl font-bold mb-2 md:mb-3 border-b-2 border-amber-600 inline-block">Support</span>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">FAQ</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Contact Us</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Help Center</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Terms of Service</Link>
                        </div>

                        {/* Find Us Section */}
                        <div className="flex flex-col items-center md:items-start text-amber-700 font-serif min-w-[120px]">
                            <span className="text-lg md:text-xl font-bold mb-2 md:mb-3 border-b-2 border-amber-600 inline-block">Find Us</span>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Events</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Locations</Link>
                            <Link className="hover:underline hover:text-amber-500 transition duration-300 text-sm md:text-base py-1">Newsletter</Link>
                        </div>

                        {/* Social Media Section */}
                        <div className="flex flex-col items-center md:items-start font-serif min-w-[140px]">
                            <span className="text-lg md:text-xl text-amber-700 font-bold mb-2 md:mb-3 border-b-2 border-amber-600 inline-block">Our Social</span>
                            <div className="flex flex-row md:flex-col gap-3 md:gap-2 mt-1">
                                <a 
                                    className="group flex items-center gap-2 text-gray-400 hover:text-pink-600 transition-all duration-300 transform hover:scale-110" 
                                    href="https://www.instagram.com/__net__250" 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M7.75 2C4.678 2 2 4.678 2 7.75v8.5C2 19.322 4.678 22 7.75 22h8.5C19.322 22 22 19.322 22 16.25v-8.5C22 4.678 19.322 2 16.25 2h-8.5zm0 2h8.5C18.216 4 20 5.784 20 7.75v8.5c0 1.966-1.784 3.75-3.75 3.75h-8.5C5.784 20 4 18.216 4 16.25v-8.5C4 5.784 5.784 4 7.75 4zm9.25 1.5a.75.75 0 100 1.5.75.75 0 000-1.5zM12 7a5 5 0 100 10 5 5 0 000-10zm0 2a3 3 0 110 6 3 3 0 010-6z"/>
                                    </svg>
                                    <span className="hidden md:inline text-sm">Instagram</span>
                                </a>
                                
                                <a 
                                    className="group flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-all duration-300 transform hover:scale-110" 
                                    href="https://x.com/Niyomugabo_250" 
                                    rel="noopener noreferrer" 
                                    target="_blank"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M18.244 2H21l-6.51 7.44L22 22h-6.828l-5.35-6.98L3.9 22H1l6.97-7.97L2 2h6.828l4.86 6.37L18.244 2zm-2.394 18h1.885L8.19 4H6.2l9.65 16z"/>
                                    </svg>
                                    <span className="hidden md:inline text-sm">Twitter/X</span>
                                </a>
                                
                                <a 
                                    className="group flex items-center gap-2 text-gray-400 hover:text-blue-700 transition-all duration-300 transform hover:scale-110" 
                                    href="https://web.facebook.com/profile.php?id=100090629463936" 
                                    rel="noopener noreferrer" 
                                    target="_blank"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M22 12a10 10 0 10-11.5 9.87v-6.99H8.078V12h2.422V9.797c0-2.39 1.42-3.713 3.596-3.713 1.043 0 2.134.186 2.134.186v2.35h-1.203c-1.185 0-1.553.736-1.553 1.49V12h2.64l-.422 2.88h-2.218v6.99A10 10 0 0022 12z"/>  
                                    </svg>
                                    <span className="hidden md:inline text-sm">Facebook</span>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Copyright Section */}
            <div className="border-t border-stone-800 mt-4 md:mt-6">
                <div className="max-w-7xl mx-auto px-4 md:px-6 lg:px-8 py-4 md:py-6">
                    <p className="text-sm md:text-base lg:text-lg text-center text-gray-300 font-serif">
                        &copy; {new Date().getFullYear()} Predictor. Built with 
                        <span className="text-red-500 inline-block animate-pulse"> ❤️ </span> 
                        and 
                        <span className="text-amber-600"> ☕ </span> 
                        by 
                        <a 
                            href="https://github.com/niyomugaboetiene" 
                            rel="noopener noreferrer" 
                            target="_blank" 
                            className="text-amber-600 hover:text-amber-500 underline font-bold transition-colors duration-300 ml-1"
                        >
                            Etiene Niyomugabo
                        </a>
                    </p>
                </div>
            </div>
        </div>
    )
}

export default Footer;