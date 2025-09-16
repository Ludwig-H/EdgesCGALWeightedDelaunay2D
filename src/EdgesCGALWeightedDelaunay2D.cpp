
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Regular_triangulation_vertex_base_2.h>
#include <CGAL/Regular_triangulation_face_base_2.h>
#include <CGAL/spatial_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_2.h>
#include <CGAL/property_map.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <utility>
#include <cctype>


namespace npy {
[[noreturn]] void die(const std::string& m){ std::fprintf(stderr,"[error] %s\n", m.c_str()); std::exit(1); }
struct Header { std::string descr; bool fortran=false; std::vector<int64_t> shape; };
static Header read_header(std::istream& in){
  Header h; char magic[6]; in.read(magic,6);
  if(!in || std::memcmp(magic,"\x93NUMPY",6)!=0) die("fichier .npy invalide (magic)");
  char ver[2]; in.read(ver,2);
  uint32_t h32=0; uint16_t h16=0; size_t hlen=0;
  if(ver[0]==1){ in.read(reinterpret_cast<char*>(&h16),2); hlen=h16; }
  else { in.read(reinterpret_cast<char*>(&h32),4); hlen=h32; }
  std::string hdr(hlen,'\0'); in.read(hdr.data(),hlen); if(!in) die("header .npy tronqué");
  bool got_descr=false; size_t search=0;
  while(!got_descr){
    auto pd=hdr.find("descr", search);
    if(pd==std::string::npos) break;
    search=pd+5;
    bool has_quote=false; char key_quote='\0';
    if(pd>0){
      char prev=hdr[pd-1];
      if(prev=='\'' || prev=='\"'){ has_quote=true; key_quote=prev; }
      else if(std::isalnum(static_cast<unsigned char>(prev)) || prev=='_') continue;
    }
    size_t after=pd+5;
    if(has_quote){
      if(after>=hdr.size() || hdr[after]!=key_quote) continue;
      ++after;
    } else if(after<hdr.size() && (std::isalnum(static_cast<unsigned char>(hdr[after])) || hdr[after]=='_')) continue;
    while(after<hdr.size() && std::isspace(static_cast<unsigned char>(hdr[after]))) ++after;
    if(after>=hdr.size() || hdr[after]!=':') continue;
    ++after;
    while(after<hdr.size() && std::isspace(static_cast<unsigned char>(hdr[after]))) ++after;
    if(after>=hdr.size()) die("header invalide (descr valeur)");
    char quote=hdr[after];
    if(quote!='\'' && quote!='\"') die("header invalide (descr valeur)");
    auto q2=hdr.find(quote, after+1);
    if(q2==std::string::npos) die("header invalide (descr valeur)");
    if(q2<=after+1) die("descr invalide");
    h.descr=hdr.substr(after+1, q2-after-1);
    got_descr=true;
  }
  if(!got_descr) die("header invalide (descr)");
  auto pf=hdr.find("fortran_order"); if(pf==std::string::npos) die("header invalide (fortran_order)"); h.fortran=hdr.find("True",pf)!=std::string::npos;
  auto ps=hdr.find("shape"); if(ps==std::string::npos) die("header invalide (shape)");
  auto lp=hdr.find('(',ps), rp=hdr.find(')',lp); if(lp==std::string::npos||rp==std::string::npos) die("shape invalide");
  std::string tup=hdr.substr(lp+1,rp-lp-1); size_t i=0; 
  while(i<tup.size()){ while(i<tup.size()&&(tup[i]==','||tup[i]==' ')) ++i; size_t j=i; while(j<tup.size()&&isdigit((unsigned char)tup[j])) ++j; if(j>i){ h.shape.push_back(std::stoll(tup.substr(i,j-i))); i=j; } else ++i; }
  return h;
}
static inline bool host_is_le(){ uint16_t x=1; return *(uint8_t*)&x==1; }
template<class T> inline void bswap(void*);
template<> inline void bswap<float>(void* p){ auto b=(uint8_t*)p; std::swap(b[0],b[3]); std::swap(b[1],b[2]); }
template<> inline void bswap<double>(void* p){ auto b=(uint8_t*)p; std::swap(b[0],b[7]); std::swap(b[1],b[6]); std::swap(b[2],b[5]); std::swap(b[3],b[4]); }
template<class OutT> std::vector<OutT> load_real(const std::string& path, std::vector<int64_t>& shape){
  std::ifstream f(path, std::ios::binary); if(!f) die("impossible d'ouvrir "+path);
  Header h = read_header(f); if(h.fortran) die("fortran_order=True non supporté");
  if(h.descr.size()<3) die("descr invalide: "+h.descr);
  const char endian=h.descr[0], code=h.descr[1], bytes=h.descr[2];
  const bool be = (endian=='>'); const bool le = (endian=='<' || endian=='=');
  size_t count=1; for(auto d: h.shape) count*=size_t(d);
  std::vector<OutT> out(count);
  if(code=='f' && bytes=='8'){
    std::vector<double> buf(count); f.read((char*)buf.data(), count*sizeof(double)); if(!f) die("corpus tronqué "+path);
    if(be && host_is_le()) for(size_t i=0;i<count;++i) bswap<double>(&buf[i]);
    for(size_t i=0;i<count;++i) out[i]=OutT(buf[i]);
  } else if(code=='f' && bytes=='4'){
    std::vector<float> buf(count); f.read((char*)buf.data(), count*sizeof(float)); if(!f) die("corpus tronqué "+path);
    if(be && host_is_le()) for(size_t i=0;i<count;++i) bswap<float>(&buf[i]);
    for(size_t i=0;i<count;++i) out[i]=OutT(buf[i]);
  } else die("dtype non supporté: "+h.descr);
  shape=h.shape; return out;
}
inline void save_u64_2col(const std::string& path, const std::vector<std::pair<uint64_t,uint64_t>>& E){
  std::string dict = "{'descr': '<u8', 'fortran_order': False, 'shape': ("+std::to_string(E.size())+", 2), }";
  while( (10 + dict.size()) % 16 != 0 ) dict.push_back(' '); dict.back()='\n';
  std::ofstream f(path, std::ios::binary); if(!f) die("impossible d'écrire "+path);
  f.write("\x93NUMPY",6); char v[2]={1,0}; f.write(v,2); uint16_t hlen=(uint16_t)dict.size(); f.write((char*)&hlen,2); f.write(dict.data(), dict.size());
  for(const auto& e: E){ uint64_t a=e.first,b=e.second; f.write((char*)&a,sizeof a); f.write((char*)&b,sizeof b); }
}
} // namespace npy
int main(int argc, char** argv){
  if(argc!=4){
    std::fprintf(stderr, "Usage: %s points.npy weights.npy out_edges.npy\n", argv[0]);
    return 64;
  }
  std::vector<int64_t> shpP, shpW;
  auto P = npy::load_real<double>(argv[1], shpP);
  auto W = npy::load_real<double>(argv[2], shpW);
  if(!(shpP.size()==2 && shpP[1]==2)) npy::die("points.npy: shape (N,2) exigée");
  size_t N = (size_t)shpP[0];
  size_t NW = (shpW.size()==1? (size_t)shpW[0] : ((shpW.size()==2 && shpW[1]==1)? (size_t)shpW[0] : 0));
  if(NW!=N) npy::die("weights.npy: shape (N,) exigée");

  using K   = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Vbb = CGAL::Regular_triangulation_vertex_base_2<K>;
  using Vb  = CGAL::Triangulation_vertex_base_with_info_2<uint64_t, K, Vbb>;
  using Fb  = CGAL::Regular_triangulation_face_base_2<K>;
  using TDS = CGAL::Triangulation_data_structure_2<Vb, Fb>;
  using RT  = CGAL::Regular_triangulation_2<K, TDS>;
  using WP = RT::Weighted_point;
  using BP = K::Point_2;

  struct Vertex_payload {
    K::FT weight;
    uint64_t index;
  };
  using Entry = std::pair<BP, Vertex_payload>;

  std::vector<Entry> entries;
  entries.reserve(N);
  for(size_t i=0;i<N;++i){
    const double* r = &P[i*2];
    entries.emplace_back(BP(r[0], r[1]), Vertex_payload{K::FT(W[i]), uint64_t(i)});
  }

  using Point_map = CGAL::First_of_pair_property_map<Entry>;
  using Sort_traits = CGAL::Spatial_sort_traits_adapter_2<K, Point_map>;
  CGAL::spatial_sort(entries.begin(), entries.end(), Sort_traits(Point_map()));

  RT rt;
  for(const auto& entry : entries){
    WP wp(entry.first, entry.second.weight); // poids = poids de puissance (souvent rayon^2)
    auto vh = rt.insert(wp);
    if(vh != RT::Vertex_handle()) vh->info() = entry.second.index;
  }

  std::vector<std::pair<uint64_t,uint64_t>> E;
  E.reserve(3*N);
  for(auto eit = rt.finite_edges_begin(); eit != rt.finite_edges_end(); ++eit){
    auto f = eit->first; int i = eit->second;
    auto va = f->vertex((i+1)%3);
    auto vb = f->vertex((i+2)%3);
    uint64_t a = va->info(), b = vb->info();
    if(a==b) continue;
    if(a<b) E.emplace_back(a,b); else E.emplace_back(b,a);
  }
  std::sort(E.begin(), E.end());
  E.erase(std::unique(E.begin(), E.end()), E.end());

  npy::save_u64_2col(argv[3], E);
  std::fprintf(stderr, "[info] N=%zu edges=%zu\n", N, E.size());
  return 0;
}