import { Outlet } from "react-router-dom";
import { Moon, Sun } from 'lucide-react'
import { useState, useEffect } from "react";

function Layout() {
    const [darkMode, setDarkMode] = useState(true);

    useEffect(() => {
        document.body.className = darkMode ? "dark" : "light";
    }, [darkMode]);

    const toggleTheme = () => {
        setDarkMode(prev => !prev);
    };
    return (
        <>
            <header>
                <section id="header-left">
                    <h3>made with &lt;3 by mzums</h3>
                </section>
                <section id="header-right">
                    <a id="header-btn" href="https://github.com/mzums/fakewiki/pulls" target="_blank" rel="noreferrer">
                        contribute
                    </a>
                    <a id="header-btn" href="https://github.com/mzums/fakewiki" target="_blank" rel="noreferrer">
                        GitHub
                    </a>
                    <button
                        onClick={toggleTheme}
                        className={`icon-btn ${darkMode ? "moon" : "sun"}`}
                    >
                        {darkMode ? <Moon /> : <Sun />}
                    </button>
                </section>
            </header>

            <main>
                <Outlet />
            </main>

            <footer>
                <p>Note that all information on this page is fictional and this project was created for educational purposes only.</p>
            </footer>

        </>
    );
}

export default Layout;