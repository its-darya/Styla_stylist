import { createFileRoute } from "@tanstack/react-router";
import { Eye } from "lucide-react";

export const Route = createFileRoute("/discover")({
  head: () => ({
    meta: [
      { title: "Discover \u2014 Styla" },
      { name: "description", content: "Get inspired by the community's fashion combinations." },
    ],
  }),
  component: DiscoverPage,
});

interface DiscoverPost {
  id: string;
  imageUrl: string;
  author: string;
  style: string;
  views: string;
}

const mockPosts: DiscoverPost[] = [
  { id: "1",  imageUrl: "https://images.pexels.com/photos/27641316/pexels-photo-27641316.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Maria",   style: "Casual Chic",     views: "2.6m" },
  { id: "2",  imageUrl: "https://images.pexels.com/photos/31046830/pexels-photo-31046830.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Abi",      style: "Minimalist",      views: "2.5m" },
  { id: "3",  imageUrl: "https://images.pexels.com/photos/29398132/pexels-photo-29398132.jpeg?auto=compress&cs=tinysrgb&w=600", author: "saisho",   style: "Y2K",             views: "1.7m" },
  { id: "4",  imageUrl: "https://images.pexels.com/photos/31046829/pexels-photo-31046829.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Mona",     style: "Streetwear",      views: "1.1m" },
  { id: "5",  imageUrl: "https://images.pexels.com/photos/31046827/pexels-photo-31046827.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Katie",    style: "Preppy",          views: "1.8m" },
  { id: "6",  imageUrl: "https://images.pexels.com/photos/30381008/pexels-photo-30381008.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Stine",    style: "Vintage",         views: "1.8m" },
  { id: "7",  imageUrl: "https://images.pexels.com/photos/31046841/pexels-photo-31046841.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Yuki",     style: "Boho",            views: "900k" },
  { id: "8",  imageUrl: "https://images.pexels.com/photos/30590661/pexels-photo-30590661.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Elena",    style: "Elegant",         views: "3.2m" },
  { id: "9",  imageUrl: "https://images.pexels.com/photos/13568592/pexels-photo-13568592.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Chloe",    style: "Business Casual", views: "1.5m" },
  { id: "10", imageUrl: "https://images.pexels.com/photos/13568611/pexels-photo-13568611.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Sofia",    style: "Edgy",            views: "2.1m" },
  { id: "11", imageUrl: "https://images.pexels.com/photos/27542890/pexels-photo-27542890.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Lina",     style: "Sporty",          views: "1.4m" },
  { id: "12", imageUrl: "https://images.pexels.com/photos/27383816/pexels-photo-27383816.jpeg?auto=compress&cs=tinysrgb&w=600", author: "Ayla",     style: "Formal",          views: "2.8m" },
];

function DiscoverPage() {
  return (
    <div className="space-y-10 pb-12">
      <header className="text-center pt-8 pb-2">
        <p className="text-xs uppercase tracking-[0.25em] text-primary">Inspiration</p>
        <h1 className="mt-4 text-4xl md:text-5xl font-display tracking-tight">Discover Styles</h1>
        <p className="mt-4 max-w-xl mx-auto text-muted-foreground text-base">
          Get inspired by the community. Find your next signature look.
        </p>
      </header>

      {/* altadaily-style grid: clean bg cards, name + views below */}
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-x-4 gap-y-8">
        {mockPosts.map((post) => (
          <div key={post.id} className="group cursor-pointer">
            {/* Image — neutral bg like altadaily lookbook */}
            <div className="relative bg-[#f3f3f3] rounded-lg overflow-hidden aspect-[3/4.5] transition-all duration-300 hover:shadow-lg hover:-translate-y-0.5">
              <img
                src={post.imageUrl}
                alt={`Outfit by ${post.author}`}
                loading="lazy"
                className="w-full h-full object-cover object-top transition-transform duration-500 group-hover:scale-[1.03]"
              />
            </div>

            {/* Info below — altadaily layout */}
            <div className="mt-2 flex items-start justify-between gap-1">
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground truncate">{post.author}</p>
                <p className="text-[11px] text-muted-foreground mt-0.5">{post.style}</p>
              </div>
              <span className="flex items-center shrink-0 text-[11px] text-muted-foreground mt-0.5">
                <Eye className="w-3 h-3 mr-0.5" />
                {post.views}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
