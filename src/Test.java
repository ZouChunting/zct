import java.util.*;

public class Test {
    public static Map<String, String> sortMapByValue(Map<String, String> oriMap) {
        if (oriMap == null || oriMap.isEmpty()) {
            return null;
        }
        Map<String, String> sortedMap = new LinkedHashMap<String, String>();
        List<Map.Entry<String, String>> entryList = new ArrayList<Map.Entry<String, String>>(oriMap.entrySet());

        Collections.sort(entryList,new Comparator<Map.Entry<String, String>>(){
            @Override
            public int compare(Map.Entry<String, String> o1, Map.Entry<String, String> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });

        Iterator<Map.Entry<String, String>> iter = entryList.iterator();
        Map.Entry<String, String> tmpEntry = null;
        while (iter.hasNext()) {
            tmpEntry = iter.next();
            sortedMap.put(tmpEntry.getKey(), tmpEntry.getValue());
        }
        return sortedMap;
    }

    public static void main(String[] args){
        /*Map<String, String> map = new TreeMap<String, String>();
        map.put("newTagCategory_027","0.922");
        map.put("newTagCategory_026","0.922");
        map.put("newTagCategory_056","0.9");
        Map<String, String> re = sortMapByValue(map);
        System.out.println(re);*/

        Zan zan = new Zan();
        zan.like(111,789);
    }
}

class Zan{
    static HashMap<String,ArrayList<String>> hashMap = new HashMap<>();

    public Zan(){
        ArrayList<String> arrayList = new ArrayList<>();
        arrayList.add("123");
        arrayList.add("456");
        hashMap.put("789",arrayList);
    }

    //点赞接口
    public void like(int uid, int statusid){
        if(!isLiked(uid,statusid)){
            hashMap.get(Integer.toString(statusid)).add(Integer.toString(uid));
            System.out.println(hashMap.get(Integer.toString(statusid)).size()+" people liked this mblog now");
        }else {
            System.out.println("you have liked this mblog");
        }
    }

    //查询赞接口
    public boolean isLiked(int uid, int statusid){
        boolean res = false;
        if(hashMap.containsKey(Integer.toString(statusid))){
            ArrayList<String> arrayList = hashMap.get(Integer.toString(statusid));
            if(arrayList.contains(Integer.toString(uid))){
                res = true;
            }
        }
        return res;
    }
}
