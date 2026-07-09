import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

interface ArticleData {
    title: string;
    sections: Record<string, string>;
}

function HomePage() {
    const [articles, setArticles] = useState<ArticleData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchArticles = async () => {
            try {
                const response = await fetch('/b.txt');
                if (!response.ok) throw new Error('Error loading file');
                const text = await response.text();
                const data = JSON.parse(text);
                const articlesArray = Array.isArray(data) ? data : [data];
                setArticles(articlesArray);
            } catch (err) {
                setError('Błąd ładowania artykułów');
            } finally {
                setLoading(false);
            }
        };
        fetchArticles();
    }, []);

    if (loading) return <div>Ładowanie...</div>;
    if (error) return <div style={{ color: 'red' }}>{error}</div>;

    return (
        <ul>
            {articles.map((article, index) => (
                <li key={index}>
                    <Link to={`/Article/${encodeURIComponent(article.title)}`}>
                        {article.title}
                    </Link>
                </li>
            ))}
        </ul>
    );
}

export default HomePage;