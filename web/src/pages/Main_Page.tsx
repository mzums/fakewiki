import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react';
import '../index.css'

interface DYK_OTD_Data {
    content: string
}
interface ArticleData {
    title: string;
    abstract: string;
}

function Main_Page() {
    const [dyk, setDyk] = useState<DYK_OTD_Data[]>([]);
    const [article, setArticle] = useState<ArticleData[]>([]);
    const [popular, setPopular] = useState<ArticleData[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        function hashInt(x: number): number {
            x = x >>> 0;
            x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
            x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
            x = x ^ (x >>> 16);
            return x >>> 0;
        }

        const dateSeed = +new Date().toISOString().slice(0, 10).replace(/-/g, '');

        const pickDYK = <T,>(arr: T[], n = 10): NonNullable<T>[] => {
            const result: NonNullable<T>[] = [];
            for (let i = 0; i < n; i++) {
                const r = hashInt(dateSeed + i) / 0xFFFFFFFF;
                const idx = Math.floor(r * arr.length);
                result.push(arr[idx] as NonNullable<T>);
            }
            return result;
        };

        const pickArticle = <T extends { title?: string; sections?: { Abstract?: string } }>(
            arr: T[],
            seed: number
        ): { title: string; abstract: string } => {
            const random = hashInt(seed) / 0xFFFFFFFF;
            const index = Math.floor(random * arr.length);
            const picked = arr[index];

            if (!picked) {
                const fallback = arr.find(item => item != null);
                if (fallback) {
                    return {
                        title: fallback.title ?? '',
                        abstract: fallback.sections?.Abstract ?? '',
                    };
                }
                return { title: '', abstract: '' };
            }

            return {
                title: picked.title ?? '',
                abstract: (picked.sections?.Abstract)?.slice(0, 1000) ?? '',
            };
        };

        const fetchDYK = async () => {
            try {
                const baseUrl = import.meta.env.BASE_URL || '/';
                const response = await fetch(`${baseUrl}dyk.json`);
                if (!response.ok) throw new Error('Error loading file');

                const text = await response.text();
                let data: DYK_OTD_Data[] = [];

                try {
                    const parsed = JSON.parse(text);
                    if (Array.isArray(parsed)) {
                        data = parsed;
                    } else if (parsed && typeof parsed === 'object' && 'title' in parsed) {
                        data = [parsed as DYK_OTD_Data];
                    } else {
                        throw new Error('Unexpected format');
                    }
                } catch {
                    console.log("Error parsing DYK")
                }

                const selected = pickDYK(data, 10);
                setDyk(selected);
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        const fetchArticles = async () => {
            try {
                const baseUrl = import.meta.env.BASE_URL || '/';
                const response = await fetch(`${baseUrl}articles.json`);
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
                } catch (err) {
                    setError(err instanceof Error ? err.message : String(err));
                } finally {
                    setLoading(false);
                }

                const selected = pickArticle(data, dateSeed)
                setArticle([selected]);
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        const fetchPopular = async () => {
            try {
                const baseUrl = import.meta.env.BASE_URL || '/';
                const response = await fetch(`${baseUrl}articles.json`);
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

                const width = window.innerWidth;
                let limit = 20;
                if (width >= 1500) {
                    limit = 30
                }

                setPopular(data.slice(0, limit));

            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        };

        fetchDYK();
        fetchArticles();
        fetchPopular();
    }, []);

    if (loading) return <div>Loading article list...</div>;
    if (error) return <div style={{ color: 'red' }}>Error: {error}</div>;

    return (
        <>
            <div className="main-container">
                <img id="main-img" src="fake2.png" alt="logo" />
                <section id="welcome">
                    <h1>Welcome to <a href="https://github.com/mzums/fakewiki">FakeWiki</a>,</h1>
                    <h3>the fake encyclopedia maintained by <a href="https://mzums.com">mzums</a></h3>
                    <h3>
                        Articles generated by a custom-trained AI model based on data from <a href="https://en.wikipedia.org">Wikipedia</a>.
                    </h3>
                </section>
            </div>



            <section id="cards">
                <h4 id="view-all">
                    view  <Link to="/All_Articles">all articles...</Link>
                </h4>
                <section id="card-row">
                    <section id="card">
                        <section id="card-title">
                            From today's featured article
                        </section>
                        {article.map((item, index) => (
                            <p key={index} id="card-text">{item.abstract}
                                <br></br>
                                <p style={{ fontStyle: "italic" }}>
                                    (
                                    <Link id="full" to={`/Article/${item.title}`}>
                                        Full article...
                                    </Link>
                                    )
                                </p>
                            </p >
                        ))}
                    </section>

                    <section id="card">
                        <section id="card-title">
                            Most popular
                        </section>
                        <ul id="popular">
                            {popular.map((article, index) => (
                                <li key={index}>
                                    <Link to={`/Article/${article.title}`}>
                                        {article.title}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </section>
                </section>

                <section id="card-row">
                    <section id="card">
                        <section id="card-title">
                            Did you know...
                        </section>
                        <ul id="dyk-list">
                            {dyk.map((dyk, index) => (
                                <li key={index}>
                                    {dyk.content}
                                </li>
                            ))}
                        </ul>
                    </section>

                </section>
            </section >
        </>
    )
}

export default Main_Page