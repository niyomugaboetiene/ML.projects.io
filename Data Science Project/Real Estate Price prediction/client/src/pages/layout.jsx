import Footer from "./footer";

const Layout = ({ children }) => {
    return (
        <div className="min-h-screen">
            <main className="pt-6">
                {children}
            </main>

            <Footer />
        </div>
    )
}

export default Layout;