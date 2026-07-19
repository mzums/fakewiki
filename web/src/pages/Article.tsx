import { useState, useEffect } from 'react';
import { useMatch } from 'react-router-dom'; // useParams didn't work

interface ArticleData {
    title: string;
    sections?: Record<string, string>;
}

function Article() {
    const match = useMatch('/Article/:title');
    const title = match?.params.title;

    const [article, setArticle] = useState<ArticleData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchArticle = async () => {
            if (!title) {
                setError('No title in address');
                setLoading(false);
                return;
            }

            try {
                const baseUrl = import.meta.env.BASE_URL || '/';
                const response = await fetch(`${baseUrl}articles.json`);
                if (!response.ok) throw new Error('Error loading file');

                const text = await response.text();
                let articles: ArticleData[] = [];

                try {
                    const parsed = JSON.parse(text);
                    if (Array.isArray(parsed)) {
                        articles = parsed;
                    } else if (parsed && typeof parsed === 'object' && 'title' in parsed) {
                        articles = [parsed as ArticleData];
                    } else {
                        throw new Error('Unexpected format');
                    }
                } catch {
                    articles = text
                        .split('\n')
                        .filter(line => line.trim() !== '')
                        .map(line => JSON.parse(line) as ArticleData);
                }

                const found = articles.find(item => item.title === title);

                if (found) {
                    setArticle(found);
                } else {
                    setError(`Didn't find article with title: "${title}"`);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        fetchArticle();
    }, [title]);

    if (loading) return <div>Loading articles...</div>;
    if (error) return <div style={{ color: 'red' }}>Error: {error}</div>;
    if (!article) return <div>Article not found.</div>;

    return (
        <div id="article-page">
            <div id="content">
                <h3>Contents:</h3>
                <ul>
                    {article.sections &&
                        Object.entries(article.sections).map(([key, _]) => (
                            <li key={key}>{key}</li>
                        ))}
                </ul>
            </div>

            <div id="article-text">
                <div className="image-frame">
                    <div className="image-caption">{title}</div>
                    <img id="article-img" src="/fakewiki/image.png" alt="Image" />
                </div>

                <h1>{article.title}</h1>
                {article.sections &&
                    Object.entries(article.sections).map(([key, content]) => (
                        <div key={key}>
                            <h2>{key}</h2>
                            <p>{content}</p>
                        </div>
                    ))}
            </div>
        </div>
    );
}

export default Article;
