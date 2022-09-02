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
 * <p>Java class for ProcessorType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="ProcessorType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;choice maxOccurs="unbounded"&gt;
 *         &lt;element name="ComputingUnits" type="{http://www.w3.org/2001/XMLSchema}int"/&gt;
 *         &lt;element name="Architecture" type="{http://www.w3.org/2001/XMLSchema}string" minOccurs="0"/&gt;
 *         &lt;element name="Speed" type="{http://www.w3.org/2001/XMLSchema}float" minOccurs="0"/&gt;
 *         &lt;element name="Type" type="{http://www.w3.org/2001/XMLSchema}string" minOccurs="0"/&gt;
 *         &lt;element name="InternalMemorySize" type="{http://www.w3.org/2001/XMLSchema}float" minOccurs="0"/&gt;
 *         &lt;element name="ProcessorProperty" type="{}ProcessorPropertyType" minOccurs="0"/&gt;
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
@XmlType(name = "ProcessorType", propOrder = {
    "computingUnitsOrArchitectureOrSpeed"
})
public class ProcessorType {

    @XmlElementRefs({
        @XmlElementRef(name = "ComputingUnits", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Architecture", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Speed", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Type", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "InternalMemorySize", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "ProcessorProperty", type = JAXBElement.class, required = false)
    })
    protected List<JAXBElement<?>> computingUnitsOrArchitectureOrSpeed;
    @XmlAttribute(name = "Name", required = true)
    protected String name;

    /**
     * Gets the value of the computingUnitsOrArchitectureOrSpeed property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the computingUnitsOrArchitectureOrSpeed property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getComputingUnitsOrArchitectureOrSpeed().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link JAXBElement }{@code <}{@link Integer }{@code >}
     * {@link JAXBElement }{@code <}{@link String }{@code >}
     * {@link JAXBElement }{@code <}{@link Float }{@code >}
     * {@link JAXBElement }{@code <}{@link String }{@code >}
     * {@link JAXBElement }{@code <}{@link Float }{@code >}
     * {@link JAXBElement }{@code <}{@link ProcessorPropertyType }{@code >}
     * 
     * 
     */
    public List<JAXBElement<?>> getComputingUnitsOrArchitectureOrSpeed() {
        if (computingUnitsOrArchitectureOrSpeed == null) {
            computingUnitsOrArchitectureOrSpeed = new ArrayList<JAXBElement<?>>();
        }
        return this.computingUnitsOrArchitectureOrSpeed;
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
