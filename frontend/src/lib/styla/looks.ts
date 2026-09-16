/**
 * Curated sample looks, shared by Discover and the Reference page.
 *
 * Each image is a free-licence studio photo with its background removed and
 * the figure placed on the same white canvas — the same visual domain as the
 * wardrobe cut-outs, which is what the reference matcher compares against.
 */
export interface Look {
  id: string;
  imageUrl: string;
  style: string;
  views: string;
}

export const LOOKS: Look[] = [
  { id: "19663779", imageUrl: "/discover/pexels-19663779.jpg", style: "Minimalist",      views: "2.4m" },
  { id: "19317147", imageUrl: "/discover/pexels-19317147.jpg", style: "Sporty",          views: "1.2m" },
  { id: "15576188", imageUrl: "/discover/pexels-15576188.jpg", style: "Y2K",             views: "2.1m" },
  { id: "16647788", imageUrl: "/discover/pexels-16647788.jpg", style: "Sporty",          views: "980k" },
  { id: "17745134", imageUrl: "/discover/pexels-17745134.jpg", style: "Casual",          views: "1.6m" },
  { id: "20664147", imageUrl: "/discover/pexels-20664147.jpg", style: "Elegant",         views: "2.7m" },
  { id: "27517706", imageUrl: "/discover/pexels-27517706.jpg", style: "Preppy",          views: "1.4m" },
  { id: "10265323", imageUrl: "/discover/pexels-10265323.jpg", style: "Streetwear",      views: "1.1m" },
  { id: "14019358", imageUrl: "/discover/pexels-14019358.jpg", style: "Sporty",          views: "870k" },
  { id: "7137421",  imageUrl: "/discover/pexels-7137421.jpg",  style: "Business Casual", views: "1.8m" },
  { id: "3888210",  imageUrl: "/discover/pexels-3888210.jpg",  style: "Minimalist",      views: "2.2m" },
  { id: "19040407", imageUrl: "/discover/pexels-19040407.jpg", style: "Edgy",            views: "1.3m" },
  { id: "19317143", imageUrl: "/discover/pexels-19317143.jpg", style: "Streetwear",      views: "760k" },
  { id: "13791573", imageUrl: "/discover/pexels-13791573.jpg", style: "Elegant",         views: "2.9m" },
  { id: "7282184",  imageUrl: "/discover/pexels-7282184.jpg",  style: "Y2K",             views: "1.5m" },
  { id: "18491439", imageUrl: "/discover/pexels-18491439.jpg", style: "Boho",            views: "690k" },
];
