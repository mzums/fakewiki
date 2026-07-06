import { useEffect, useState } from "react";
import { Link } from 'react-router-dom'


function All_Articles() {
    const [lines, setLines] = useState<string[]>([]);

    useEffect(() => {
        fetch("/a.txt")
            .then(response => response.text())
            .then(text => setLines(text.split("\n")));
    }, []);

    return (
        <>
            <h1 className="article-title">
                List of all articles
            </h1>
            <ul>
                {lines.map((line, index) => (

                    <li key={index}>
                        <Link to="/All_Articles">{line}</Link>
                    </li>
                ))}
            </ul>
        </>
    );
}

export default All_Articles;