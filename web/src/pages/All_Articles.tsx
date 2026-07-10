import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

interface ArticleData {
    title: string;
    sections?: Record<string, string>;
}

function AllArticles() {
    const [articles, setArticles] = useState<ArticleData[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchArticles = async () => {
            try {
                const response = await fetch('/articles.json');
                if (!response.ok) throw new Error('Error loading file');

                const text = await response.text();
                let data: ArticleData[] = [];

                try {
                    const parsed = JSON.parse(text);
                    if (Array.isArray(parsed)) {
                        data = parsed;
                    } else if (parsed && typeof parsed === 'object' && 'title' in parsed) {
                        data = [parsed as ArticleData];
                    } else {
                        throw new Error('Unexpected format');
                    }
                } catch {
                    data = text
                        .split('\n')
                        .filter(line => line.trim() !== '')
                        .map(line => JSON.parse(line) as ArticleData);
                }

                setArticles(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        fetchArticles();
    }, []);

    if (loading) return <div>Loading article list...</div>;
    if (error) return <div style={{ color: 'red' }}>Błąd: {error}</div>;

    return (
        <div id="article-list">
            <h1>Article List</h1>
            <ul>
                {articles.map((article, index) => (
                    <li key={index}>
                        <Link to={`/Article/${article.title}`}>
                            {article.title}
                        </Link>
                    </li>
                ))}
            </ul>
        </div>
    );
}

export default AllArticles;