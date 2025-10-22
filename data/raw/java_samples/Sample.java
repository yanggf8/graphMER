import java.util.List;
import java.util.ArrayList;

public class DataProcessor {
    private List<String> items;
    
    public DataProcessor() {
        this.items = new ArrayList<>();
    }
    
    public void addItem(String item) {
        items.add(item);
        validateItem(item);
    }
    
    private boolean validateItem(String item) {
        return item != null && item.length() > 0;
    }
    
    public List<String> getItems() {
        return items;
    }
}
