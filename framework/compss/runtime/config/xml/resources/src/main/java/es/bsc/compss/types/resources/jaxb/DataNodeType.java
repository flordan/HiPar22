//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.3.0 
// See <a href="https://javaee.github.io/jaxb-v2/">https://javaee.github.io/jaxb-v2/</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2022.08.23 at 06:04:07 AM CEST 
//


package es.bsc.compss.types.resources.jaxb;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElementRef;
import javax.xml.bind.annotation.XmlElementRefs;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for DataNodeType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="DataNodeType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;choice maxOccurs="unbounded" minOccurs="3"&gt;
 *         &lt;element name="Host" type="{http://www.w3.org/2001/XMLSchema}string"/&gt;
 *         &lt;element name="Path" type="{http://www.w3.org/2001/XMLSchema}string"/&gt;
 *         &lt;element name="Adaptors" type="{}AdaptorsListType"/&gt;
 *         &lt;element name="Storage" type="{}StorageType" minOccurs="0"/&gt;
 *         &lt;element name="SharedDisks" type="{}AttachedDisksListType" minOccurs="0"/&gt;
 *       &lt;/choice&gt;
 *       &lt;attribute name="Name" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "DataNodeType", propOrder = {
    "hostOrPathOrAdaptors"
})
public class DataNodeType {

    @XmlElementRefs({
        @XmlElementRef(name = "Host", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Path", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Adaptors", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Storage", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "SharedDisks", type = JAXBElement.class, required = false)
    })
    protected List<JAXBElement<?>> hostOrPathOrAdaptors;
    @XmlAttribute(name = "Name", required = true)
    protected String name;

    /**
     * Gets the value of the hostOrPathOrAdaptors property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the hostOrPathOrAdaptors property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getHostOrPathOrAdaptors().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link JAXBElement }{@code <}{@link String }{@code >}
     * {@link JAXBElement }{@code <}{@link String }{@code >}
     * {@link JAXBElement }{@code <}{@link AdaptorsListType }{@code >}
     * {@link JAXBElement }{@code <}{@link StorageType }{@code >}
     * {@link JAXBElement }{@code <}{@link AttachedDisksListType }{@code >}
     * 
     * 
     */
    public List<JAXBElement<?>> getHostOrPathOrAdaptors() {
        if (hostOrPathOrAdaptors == null) {
            hostOrPathOrAdaptors = new ArrayList<JAXBElement<?>>();
        }
        return this.hostOrPathOrAdaptors;
    }

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

}
