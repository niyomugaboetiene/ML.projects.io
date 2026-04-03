
const Layout = ({ children }) => {
    return (
        <div className="min-h-screen">
            <main className="pt-6">
                {children}
            </main>
        </div>
    )
}

export default Layout;